import argparse
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model.image_model import DMCI
from model.video_model import DMC


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def pad_to_multiple(frames: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """Pad a tensor of shape (T, C, H, W) to multiples of a given value using replicate padding."""

    if multiple <= 1:
        return frames

    _, _, height, width = frames.shape
    target_height = (height + multiple - 1) // multiple * multiple
    target_width = (width + multiple - 1) // multiple * multiple

    pad_bottom = target_height - height
    pad_right = target_width - width

    if pad_bottom == 0 and pad_right == 0:
        return frames

    return F.pad(frames, (0, pad_right, 0, pad_bottom), mode="replicate")


class Video90kDataset(Dataset):
    """Dataset loader for Video-90k sequences producing luminance GOP tensors."""

    def __init__(
        self,
        root_dir: str,
        list_file: str,
        crop_size: Optional[int] = 256,
        training: bool = True,
        seq_length: int = 7,
        padding_multiple: int = 16,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.training = training
        self.seq_length = seq_length
        self.padding_multiple = padding_multiple

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Video-90k root directory not found: {self.root_dir}")

        list_path = Path(list_file)
        if not list_path.exists():
            raise FileNotFoundError(f"Video-90k list file not found: {list_path}")

        with list_path.open("r", encoding="utf-8") as handle:
            self.sequence_ids = [line.strip() for line in handle if line.strip()]

        if len(self.sequence_ids) == 0:
            raise ValueError(f"No sequences found in list file: {list_path}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.sequence_ids)

    def _load_sequence_frames(self, sequence_id: str) -> List[Image.Image]:
        frames: List[Image.Image] = []
        # Video-90k indices start at 1 and end at 7 for each septuplet.
        upper = 1 + self.seq_length
        for index in range(1, upper):
            frame_path = self.root_dir / sequence_id / f"im{index}.png"
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame not found: {frame_path}")
            # Convert to Y (luminance) to match single-channel DCVC-RT models
            y_channel = Image.open(frame_path).convert("YCbCr").split()[0]
            frames.append(y_channel)
        return frames

    def _augment(self, frames: Sequence[Image.Image]) -> Sequence[Image.Image]:
        if not self.training:
            return frames

        if random.random() < 0.5:
            frames = [frame.transpose(Image.FLIP_LEFT_RIGHT) for frame in frames]
        if random.random() < 0.5:
            frames = [frame.transpose(Image.FLIP_TOP_BOTTOM) for frame in frames]
        return frames

    def _crop(self, frames: Sequence[Image.Image]) -> Sequence[Image.Image]:
        if self.crop_size is None:
            return frames

        width, height = frames[0].size
        crop_w = min(self.crop_size, width)
        crop_h = min(self.crop_size, height)

        if self.training:
            x = random.randint(0, width - crop_w) if width > crop_w else 0
            y = random.randint(0, height - crop_h) if height > crop_h else 0
        else:
            x = (width - crop_w) // 2
            y = (height - crop_h) // 2

        return [frame.crop((x, y, x + crop_w, y + crop_h)) for frame in frames]

    def __getitem__(self, index: int) -> torch.Tensor:  # type: ignore[override]
        sequence_id = self.sequence_ids[index]
        frames = self._load_sequence_frames(sequence_id)
        frames = self._augment(frames)
        frames = self._crop(frames)

        tensor_frames = [self.to_tensor(frame) for frame in frames]
        clip = torch.stack(tensor_frames, dim=0)  # (T, 1, H, W)
        clip = pad_to_multiple(clip, self.padding_multiple)
        return clip


q_index_to_lambda = {i: 1 + 24 * (i - 1) for i in range(1, 65)}


def get_distortion(target: torch.Tensor, recon: torch.Tensor, criterion: torch.nn.Module) -> torch.Tensor:
    return criterion(target, recon)


def get_bpp(ref_tensor: torch.Tensor, likelihoods: torch.Tensor) -> torch.Tensor:
    size_est = (-np.log(2.0) * ref_tensor.numel())
    return torch.sum(torch.log(likelihoods)) / size_est


def compute_metrics_from_mse(mse: float) -> float:
    if mse <= 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, torch.nn.DataParallel) else module


def train_one_epoch(
    i_frame_model: torch.nn.Module,
    p_frame_model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    frame_weights: Sequence[float],
) -> dict:
    i_frame_model.train()
    p_frame_model.train()

    mse_criterion = torch.nn.MSELoss(reduction="mean")

    total_loss = 0.0
    total_mse = 0.0
    total_bpp = 0.0
    total_frames = 0
    total_sequences = 0

    progress = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)

    for batch in progress:
        batch = batch.to(device)
        batch_size, seq_len, _, _, _ = batch.shape

        q = random.randint(1, 63)
        optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0
        mse_accum = 0.0
        bpp_accum = 0.0

        feature = None

        for frame_idx in range(seq_len):
            frame = batch[:, frame_idx, :, :, :]

            if frame_idx == 0:
                x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods = i_frame_model(frame, q)
            elif frame_idx == 1:
                x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = p_frame_model(
                    frame, x_hat, None, q
                )
            else:
                x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = p_frame_model(
                    frame, None, feature, q
                )

            bit_cost = get_bpp(y_hat, y_likelihoods) + get_bpp(z_hat, z_likelihoods)
            distortion = get_distortion(frame, x_hat, mse_criterion)

            weight = frame_weights[frame_idx % len(frame_weights)]
            loss_accum += q_index_to_lambda[q] * distortion * weight + bit_cost

            mse_accum += distortion.item() * batch_size
            bpp_accum += bit_cost.item() * batch_size

        loss = loss_accum / seq_len
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_mse += mse_accum
        total_bpp += bpp_accum
        total_frames += batch_size * seq_len
        total_sequences += batch_size

        avg_loss = total_loss / max(total_sequences, 1)
        avg_mse = total_mse / max(total_frames, 1)
        avg_psnr = compute_metrics_from_mse(avg_mse)
        avg_bpp = total_bpp / max(total_frames, 1)

        progress.set_postfix(
            loss=f"{avg_loss:.4f}", mse=f"{avg_mse:.6f}", psnr=f"{avg_psnr:.2f}", bpp=f"{avg_bpp:.4f}"
        )

    avg_loss = total_loss / max(total_sequences, 1)
    avg_mse = total_mse / max(total_frames, 1)
    avg_psnr = compute_metrics_from_mse(avg_mse)
    avg_bpp = total_bpp / max(total_frames, 1)

    return {"loss": avg_loss, "mse": avg_mse, "psnr": avg_psnr, "bpp": avg_bpp}


def validate(
    i_frame_model: torch.nn.Module,
    p_frame_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    frame_weights: Sequence[float],
    q_list: Sequence[int],
) -> dict:
    i_frame_model.eval()
    p_frame_model.eval()

    mse_criterion = torch.nn.MSELoss(reduction="mean")

    total_loss = 0.0
    total_mse = 0.0
    total_bpp = 0.0
    total_frames = 0
    total_sequences = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validate", leave=False)):
            batch = batch.to(device)
            batch_size, seq_len, _, _, _ = batch.shape

            q = q_list[batch_idx % len(q_list)]

            loss_accum = 0.0
            mse_accum = 0.0
            bpp_accum = 0.0

            feature = None

            for frame_idx in range(seq_len):
                frame = batch[:, frame_idx, :, :, :]

                if frame_idx == 0:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods = i_frame_model(frame, q)
                elif frame_idx == 1:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = p_frame_model(
                        frame, x_hat, None, q
                    )
                else:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = p_frame_model(
                        frame, None, feature, q
                    )

                bit_cost = get_bpp(y_hat, y_likelihoods) + get_bpp(z_hat, z_likelihoods)
                distortion = get_distortion(frame, x_hat, mse_criterion)

                weight = frame_weights[frame_idx % len(frame_weights)]
                loss_accum += q_index_to_lambda[q] * distortion * weight + bit_cost

                mse_accum += distortion.item() * batch_size
                bpp_accum += bit_cost.item() * batch_size

            total_loss += (loss_accum / seq_len).item() * batch_size
            total_mse += mse_accum
            total_bpp += bpp_accum
            total_frames += batch_size * seq_len
            total_sequences += batch_size

    avg_loss = total_loss / max(total_sequences, 1)
    avg_mse = total_mse / max(total_frames, 1)
    avg_psnr = compute_metrics_from_mse(avg_mse)
    avg_bpp = total_bpp / max(total_frames, 1)

    return {"loss": avg_loss, "mse": avg_mse, "psnr": avg_psnr, "bpp": avg_bpp}


def save_checkpoint(
    epoch: int,
    i_frame_model: torch.nn.Module,
    p_frame_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    path: Path,
    best_metric: Optional[float] = None,
) -> None:
    state = {
        "epoch": epoch,
        "i_frame": unwrap_module(i_frame_model).state_dict(),
        "p_frame": unwrap_module(p_frame_model).state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    if best_metric is not None:
        state["best_val"] = best_metric

    torch.save(state, path)


def load_checkpoint(
    path: Path,
    i_frame_model: torch.nn.Module,
    p_frame_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
) -> Tuple[int, Optional[float]]:
    checkpoint = torch.load(path, map_location="cpu")

    unwrap_module(i_frame_model).load_state_dict(checkpoint["i_frame"])
    unwrap_module(p_frame_model).load_state_dict(checkpoint["p_frame"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    best_val = checkpoint.get("best_val")
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch, best_val


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCVC-RT on Video-90k with CompressAI entropy coding")
    parser.add_argument("--train-root", type=str, required=True, help="Path to Video-90k training frames root")
    parser.add_argument("--train-list", type=str, required=True, help="Path to Video-90k training list file")
    parser.add_argument("--val-root", type=str, help="Path to Video-90k validation frames root")
    parser.add_argument("--val-list", type=str, help="Path to Video-90k validation list file")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--step-size", type=int, default=50, help="StepLR step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="StepLR gamma")
    parser.add_argument("--crop-size", type=int, default=256, help="Random crop size")
    parser.add_argument("--padding-multiple", type=int, default=16, help="Spatial padding multiple")
    parser.add_argument("--val-interval", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--q-eval", type=str, default="15,31,63", help="Comma separated Q indices for validation")
    parser.add_argument("--gpu-ids", type=str, default=None, help="Comma separated GPU ids for DataParallel")
    parser.add_argument("--output-dir", type=str, default="logs_video", help="Directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = Path(args.output_dir) / "dcvcrt_video_latest.pth"
    best_path = Path(args.output_dir) / "dcvcrt_video_best.pth"

    val_root = args.val_root if args.val_root else args.train_root
    val_list = args.val_list if args.val_list else args.train_list

    train_dataset = Video90kDataset(
        root_dir=args.train_root,
        list_file=args.train_list,
        crop_size=args.crop_size,
        training=True,
        seq_length=7,
        padding_multiple=args.padding_multiple,
    )

    val_dataset = Video90kDataset(
        root_dir=val_root,
        list_file=val_list,
        crop_size=args.crop_size,
        training=False,
        seq_length=7,
        padding_multiple=args.padding_multiple,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    frame_weights = [0.5, 1.2, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9]

    inet = DMCI()
    pnet = DMC()

    if args.gpu_ids is not None:
        gpu_ids = [int(idx) for idx in args.gpu_ids.split(",")]
        if len(gpu_ids) == 0:
            raise ValueError("--gpu-ids must contain at least one GPU index")
        device = torch.device(f"cuda:{gpu_ids[0]}")
        inet = torch.nn.DataParallel(inet.to(device), device_ids=gpu_ids)
        pnet = torch.nn.DataParallel(pnet.to(device), device_ids=gpu_ids)
    else:
        inet = inet.to(device)
        pnet = pnet.to(device)

    optimizer = torch.optim.Adam(
        list(inet.parameters()) + list(pnet.parameters()),
        lr=args.learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    start_epoch = 0
    best_metric: Optional[float] = None

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_metric = load_checkpoint(resume_path, inet, pnet, optimizer, scheduler)
        print(f"Resumed training from {resume_path} at epoch {start_epoch}")

    q_eval = [int(q.strip()) for q in args.q_eval.split(",") if q.strip()]
    if len(q_eval) == 0:
        raise ValueError("--q-eval must specify at least one integer quality index")

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{args.epochs} -- LR: {current_lr:.6e}")

        train_stats = train_one_epoch(inet, pnet, train_loader, optimizer, device, epoch + 1, frame_weights)
        print(
            f"Train | Loss: {train_stats['loss']:.4f} | MSE: {train_stats['mse']:.6f} | "
            f"PSNR: {train_stats['psnr']:.2f} | BPP: {train_stats['bpp']:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % args.val_interval == 0:
            val_stats = validate(inet, pnet, val_loader, device, frame_weights, q_eval)
            print(
                f"Val   | Loss: {val_stats['loss']:.4f} | MSE: {val_stats['mse']:.6f} | "
                f"PSNR: {val_stats['psnr']:.2f} | BPP: {val_stats['bpp']:.4f}"
            )

            save_checkpoint(epoch, inet, pnet, optimizer, scheduler, checkpoint_path, best_metric)

            metric = val_stats["loss"]
            if best_metric is None or metric < best_metric:
                best_metric = metric
                print(f"New best validation loss: {best_metric:.6f}")
                save_checkpoint(epoch, inet, pnet, optimizer, scheduler, best_path, best_metric)
        else:
            save_checkpoint(epoch, inet, pnet, optimizer, scheduler, checkpoint_path, best_metric)

    print("Training completed.")


if __name__ == "__main__":
    main()

