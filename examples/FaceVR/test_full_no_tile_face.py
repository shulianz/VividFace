import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
import glob
import argparse
import cv2
import torch
from torchvision.transforms import v2
from einops import rearrange

from diffsynth import ModelManager, WanVideoOneStepPipeline


class FaceVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, height=None, width=None):
        self.video_dirs = [
            os.path.join(video_path, d)
            for d in sorted(os.listdir(video_path))
            if os.path.isdir(os.path.join(video_path, d))
        ]
        self.height = height
        self.width = width
        self.frame_norm = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def load_frames_from_folder(self, folder_path):
        image_files = sorted(
            glob.glob(os.path.join(folder_path, "*.png")) +
            glob.glob(os.path.join(folder_path, "*.jpg")) +
            glob.glob(os.path.join(folder_path, "*.jpeg"))
        )
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {folder_path}")

        frames = []
        for img_path in image_files:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image: {img_path}")
            img = img.astype("float32") / 255.0

            h, w = img.shape[:2]
            if self.height is not None and self.width is not None:
                if h != self.height or w != self.width:
                    print(f"Bicubic resizing: {img_path} from ({h},{w}) to ({self.height},{self.width})")
                    img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

            tensor = torch.from_numpy(img).permute(2, 0, 1)[[2, 1, 0], :, :]  # BGR -> RGB
            tensor = self.frame_norm(tensor)
            frames.append(tensor)

        video = torch.stack(frames, dim=0)          # [T, C, H, W]
        video = rearrange(video, "T C H W -> C T H W")
        return video

    def __getitem__(self, index):
        video_dir = self.video_dirs[index]
        lr_video = self.load_frames_from_folder(video_dir)
        return {
            "text": "",
            "LRvideo": lr_video,
            "path": video_dir,
        }

    def __len__(self):
        return len(self.video_dirs)


def padding(video: torch.Tensor, n: int = 16):
    _, _, _, h, w = video.shape
    pad_h = (n - h % n) % n
    pad_w = (n - w % n) % n
    if pad_h != 0 or pad_w != 0:
        print(f"Input cannot be divided by {n}, padding: pad_w={pad_w}, pad_h={pad_h}")
        video = torch.nn.functional.pad(
            video, (0, pad_w, 0, pad_h, 0, 0, 0, 0, 0, 0), mode="constant", value=0
        )
    return video, pad_h, pad_w


def unpadding(video: torch.Tensor, pad_h: int, pad_w: int, scale: int = 1):
    if pad_h != 0:
        video = video[..., :-(pad_h * scale), :]
    if pad_w != 0:
        video = video[..., :-(pad_w * scale)]
    return video


def repeat_last_frame_to_multiple_plus1(tensor: torch.Tensor, multiple: int):
    _, _, t, _, _ = tensor.shape
    if (t - 1) % multiple == 0:
        return tensor, 0

    target_t = ((t - 1) // multiple + 1) * multiple + 1
    pad = target_t - t
    if pad == 0:
        return tensor, 0

    last_frame = tensor[:, :, -1:, :, :].repeat(1, 1, pad, 1, 1)
    tensor = torch.cat([tensor, last_frame], dim=2)
    return tensor, pad


def remove_repeated_frames(tensor: torch.Tensor, pad: int):
    if pad == 0:
        return tensor
    return tensor[:, :, :-pad, :, :]


def save_frames(frames, save_dir: str, filename_pattern: str):
    os.makedirs(save_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        if filename_pattern == "000*":
            filename = f"{idx:04d}.png"
        elif filename_pattern == "0000*":
            filename = f"{idx:05d}.png"
        elif filename_pattern == "0000000*":
            filename = f"{idx:08d}.png"
        else:
            filename = f"{idx:04d}.png"
        frame.save(os.path.join(save_dir, filename))


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Wan one-step face video restoration.")

    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to trained DiT checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        required=True,
        help="Path to Wan text encoder checkpoint.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="Path to Wan VAE checkpoint.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input root folder. Each subfolder is treated as one video clip of frames.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output root folder.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Resize input frames to this height. Set to -1 to disable resize.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Resize input frames to this width. Set to -1 to disable resize.",
    )
    parser.add_argument(
        "--filename_pattern",
        type=str,
        default="0000000*",
        choices=["000*", "0000*", "0000000*"],
        help="Saved frame filename pattern.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size height for VAE encode/decode.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size width for VAE encode/decode.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride height for VAE encode/decode.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride width for VAE encode/decode.",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Enable tiled VAE encode/decode.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="CFG scale for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--padding_multiple",
        type=int,
        default=16,
        help="Spatial padding multiple.",
    )
    parser.add_argument(
        "--temporal_multiple",
        type=int,
        default=4,
        help="Temporal multiple used for repeat-last-frame padding.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    height = None if args.height == -1 else args.height
    width = None if args.width == -1 else args.width

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
    model_manager.load_models([
        args.model_ckpt,
        args.text_encoder_path,
        args.vae_path,
    ])

    pipe = WanVideoOneStepPipeline.from_model_manager(model_manager, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)

    tiler_kwargs = {
        "tiled": args.tiled,
        "tile_size": (args.tile_size_height, args.tile_size_width),
        "tile_stride": (args.tile_stride_height, args.tile_stride_width),
    }

    dataset = FaceVideoDataset(
        video_path=args.input_path,
        height=height,
        width=width,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            lr_video = batch["LRvideo"].to("cuda").to(torch.bfloat16)  # [B, C, T, H, W]
            video_dir = batch["path"][0]
            video_name = os.path.basename(video_dir)
            text = batch["text"]

            lr_video, pad_h, pad_w = padding(lr_video, n=args.padding_multiple)
            lr_video, pad_t = repeat_last_frame_to_multiple_plus1(lr_video, multiple=args.temporal_multiple)

            print(f"[{batch_idx}] Input shape after padding: {tuple(lr_video.shape)}")

            pipe.device = lr_video.device
            pipe.load_models_to_device(["text_encoder", "vae", "dit"])
            prompt_emb = pipe.encode_prompt(text, positive=True)

            _, _, t, h, w = lr_video.shape

            latent = pipe.encode_video(lr_video, **tiler_kwargs)
            latent = latent.to("cuda").to(torch.bfloat16)

            restored_latent = pipe(
                prompt=text,
                negative_prompt="...",
                num_inference_steps=50,
                LR=latent,
                prompt_emb=prompt_emb,
                seed=args.seed,
                tiled=False,
                height=h * 8,
                width=w * 8,
                num_frames=t,
                cfg_scale=args.cfg_scale,
            )

            pipe.load_models_to_device(["vae"])
            restored_video = pipe.decode_video(restored_latent, **tiler_kwargs)

            restored_video = remove_repeated_frames(restored_video, pad_t)
            restored_video = restored_video[0]
            restored_video = unpadding(restored_video, pad_h, pad_w)

            print(f"[{batch_idx}] Output shape: {tuple(restored_video.shape)}")

            frames = pipe.tensor2video(restored_video)
            save_dir = os.path.join(args.output_path, video_name)
            save_frames(frames, save_dir, args.filename_pattern)

            print(f"[{batch_idx}] Saved to: {save_dir}")

    print("Inference finished.")


if __name__ == "__main__":
    main()