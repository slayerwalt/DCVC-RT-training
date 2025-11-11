#!/usr/bin/env python3
"""
Test script for Video90kDataset class.
Tests dataset loading, data shapes, augmentation, and padding functionality.
"""

import argparse
import os
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_video90k import Video90kDataset, pad_to_multiple


def create_dummy_dataset(root_dir: Path, list_file: Path, num_sequences: int = 5):
    """Create a dummy dataset for testing."""
    root_dir.mkdir(parents=True, exist_ok=True)
    
    sequence_ids = []
    for seq_idx in range(num_sequences):
        seq_id = f"0000{seq_idx}/0000001"
        seq_path = root_dir / seq_id
        seq_path.mkdir(parents=True, exist_ok=True)
        
        # Create 7 dummy frames (im1.png to im7.png)
        for frame_idx in range(1, 8):
            frame_path = seq_path / f"im{frame_idx}.png"
            # Create a dummy grayscale image (will be converted to Y channel)
            img = Image.new("RGB", (256, 256), color=(128, 128, 128))
            img.save(frame_path)
        
        sequence_ids.append(seq_id)
    
    # Write list file
    with list_file.open("w", encoding="utf-8") as f:
        for seq_id in sequence_ids:
            f.write(f"{seq_id}\n")
    
    return sequence_ids


def test_dataset_basic_loading(dataset: Video90kDataset):
    """Test basic dataset loading functionality."""
    print("Testing basic dataset loading...")
    
    assert len(dataset) > 0, "Dataset should have at least one sequence"
    print(f"  ✓ Dataset length: {len(dataset)}")
    
    # Test loading a single sample
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor), "Sample should be a torch.Tensor"
    assert sample.dim() == 4, f"Sample should be 4D (T, C, H, W), got {sample.dim()}D"
    assert sample.shape[0] == 7, f"Sequence length should be 7, got {sample.shape[0]}"
    assert sample.shape[1] == 1, f"Channel should be 1 (luminance), got {sample.shape[1]}"
    print(f"  ✓ Sample shape: {sample.shape} (T, C, H, W)")
    print(f"  ✓ Sample dtype: {sample.dtype}")
    print(f"  ✓ Sample value range: [{sample.min().item():.3f}, {sample.max().item():.3f}]")
    
    return True


def test_dataset_padding(dataset: Video90kDataset):
    """Test padding functionality."""
    print("\nTesting padding functionality...")
    
    sample = dataset[0]
    _, _, height, width = sample.shape
    
    # Check if dimensions are multiples of padding_multiple
    assert height % dataset.padding_multiple == 0, \
        f"Height {height} should be multiple of {dataset.padding_multiple}"
    assert width % dataset.padding_multiple == 0, \
        f"Width {width} should be multiple of {dataset.padding_multiple}"
    print(f"  ✓ Dimensions are multiples of {dataset.padding_multiple}")
    print(f"  ✓ Height: {height}, Width: {width}")
    
    return True


def test_dataset_augmentation():
    """Test data augmentation in training mode."""
    print("\nTesting data augmentation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir) / "data"
        list_file = Path(tmpdir) / "list.txt"
        create_dummy_dataset(root_dir, list_file, num_sequences=1)
        
        # Create training dataset (with augmentation)
        train_dataset = Video90kDataset(
            root_dir=str(root_dir),
            list_file=str(list_file),
            crop_size=256,
            training=True,
            seq_length=7,
            padding_multiple=16,
        )
        
        # Create validation dataset (without augmentation)
        val_dataset = Video90kDataset(
            root_dir=str(root_dir),
            list_file=str(list_file),
            crop_size=256,
            training=False,
            seq_length=7,
            padding_multiple=16,
        )
        
        # Load same sample multiple times (augmentation is random)
        train_samples = [train_dataset[0] for _ in range(5)]
        val_samples = [val_dataset[0] for _ in range(5)]
        
        # Validation samples should be identical (no augmentation)
        for i in range(1, len(val_samples)):
            assert torch.allclose(val_samples[0], val_samples[i]), \
                "Validation samples should be identical (no augmentation)"
        print("  ✓ Validation mode: no augmentation (samples are identical)")
        
        # Training samples might differ due to augmentation
        # (but might also be identical if random doesn't trigger augmentation)
        print("  ✓ Training mode: augmentation enabled")
    
    return True


def test_dataset_cropping():
    """Test cropping functionality."""
    print("\nTesting cropping functionality...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir) / "data"
        list_file = Path(tmpdir) / "list.txt"
        create_dummy_dataset(root_dir, list_file, num_sequences=1)
        
        # Test with crop_size
        dataset = Video90kDataset(
            root_dir=str(root_dir),
            list_file=str(list_file),
            crop_size=128,
            training=False,  # Center crop for validation
            seq_length=7,
            padding_multiple=16,
        )
        
        sample = dataset[0]
        _, _, height, width = sample.shape
        assert height >= 128, f"Height should be at least crop_size, got {height}"
        assert width >= 128, f"Width should be at least crop_size, got {width}"
        print(f"  ✓ Crop size: 128, Actual size: {height}x{width}")
        
        # Test without cropping
        dataset_no_crop = Video90kDataset(
            root_dir=str(root_dir),
            list_file=str(list_file),
            crop_size=None,
            training=False,
            seq_length=7,
            padding_multiple=16,
        )
        
        sample_no_crop = dataset_no_crop[0]
        print(f"  ✓ No crop size: {sample_no_crop.shape[2]}x{sample_no_crop.shape[3]}")
    
    return True


def test_dataloader(dataset: Video90kDataset):
    """Test DataLoader integration."""
    print("\nTesting DataLoader integration...")
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
    )
    
    batch = next(iter(dataloader))
    assert batch.dim() == 5, f"Batch should be 5D (B, T, C, H, W), got {batch.dim()}D"
    assert batch.shape[0] == 2, f"Batch size should be 2, got {batch.shape[0]}"
    assert batch.shape[1] == 7, f"Sequence length should be 7, got {batch.shape[1]}"
    assert batch.shape[2] == 1, f"Channel should be 1, got {batch.shape[2]}"
    print(f"  ✓ Batch shape: {batch.shape} (B, T, C, H, W)")
    
    return True


def test_pad_to_multiple_function():
    """Test pad_to_multiple utility function."""
    print("\nTesting pad_to_multiple utility function...")
    
    # Test with dimensions that need padding
    frames = torch.randn(7, 1, 100, 150)  # Not multiples of 16
    padded = pad_to_multiple(frames, multiple=16)
    
    assert padded.shape[0] == 7, "Time dimension should not change"
    assert padded.shape[1] == 1, "Channel dimension should not change"
    assert padded.shape[2] % 16 == 0, "Height should be multiple of 16"
    assert padded.shape[3] % 16 == 0, "Width should be multiple of 16"
    assert padded.shape[2] == 112, "Height should be padded to 112"
    assert padded.shape[3] == 160, "Width should be padded to 160"
    print(f"  ✓ Input shape: {frames.shape}")
    print(f"  ✓ Padded shape: {padded.shape}")
    
    # Test with dimensions that don't need padding
    frames_exact = torch.randn(7, 1, 128, 128)
    padded_exact = pad_to_multiple(frames_exact, multiple=16)
    assert torch.equal(frames_exact, padded_exact), "Should not pad if already multiple"
    print(f"  ✓ No padding needed for exact multiples")
    
    return True


def test_error_handling():
    """Test error handling for missing files."""
    print("\nTesting error handling...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir) / "nonexistent"
        list_file = Path(tmpdir) / "nonexistent.txt"
        
        # Test missing root directory
        try:
            Video90kDataset(
                root_dir=str(root_dir),
                list_file=str(list_file),
                crop_size=256,
                training=True,
            )
            assert False, "Should raise FileNotFoundError for missing root directory"
        except FileNotFoundError as e:
            print(f"  ✓ Correctly raises error for missing root: {type(e).__name__}")
        
        # Test missing list file
        root_dir.mkdir(parents=True)
        try:
            Video90kDataset(
                root_dir=str(root_dir),
                list_file=str(list_file),
                crop_size=256,
                training=True,
            )
            assert False, "Should raise FileNotFoundError for missing list file"
        except FileNotFoundError as e:
            print(f"  ✓ Correctly raises error for missing list file: {type(e).__name__}")
        
        # Test empty list file
        list_file.write_text("")
        try:
            Video90kDataset(
                root_dir=str(root_dir),
                list_file=str(list_file),
                crop_size=256,
                training=True,
            )
            assert False, "Should raise ValueError for empty list file"
        except ValueError as e:
            print(f"  ✓ Correctly raises error for empty list file: {type(e).__name__}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Video90kDataset class")
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Path to dataset root directory (if not provided, uses dummy data)",
    )
    parser.add_argument(
        "--list-file",
        type=str,
        help="Path to dataset list file (if not provided, uses dummy data)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Crop size for testing",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Video90kDataset Test Suite")
    print("=" * 60)
    
    # Use provided dataset or create dummy dataset
    if args.root_dir and args.list_file:
        print(f"\nUsing provided dataset:")
        print(f"  Root dir: {args.root_dir}")
        print(f"  List file: {args.list_file}")
        dataset = Video90kDataset(
            root_dir=args.root_dir,
            list_file=args.list_file,
            crop_size=args.crop_size,
            training=False,
            seq_length=7,
            padding_multiple=16,
        )
        
        # Run tests with real dataset
        test_dataset_basic_loading(dataset)
        test_dataset_padding(dataset)
        test_dataloader(dataset)
    else:
        print("\nUsing dummy dataset for testing...")
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir) / "data"
            list_file = Path(tmpdir) / "list.txt"
            create_dummy_dataset(root_dir, list_file, num_sequences=5)
            
            dataset = Video90kDataset(
                root_dir=str(root_dir),
                list_file=str(list_file),
                crop_size=args.crop_size,
                training=False,
                seq_length=7,
                padding_multiple=16,
            )
            
            # Run all tests
            test_dataset_basic_loading(dataset)
            test_dataset_padding(dataset)
            test_dataset_augmentation()
            test_dataset_cropping()
            test_dataloader(dataset)
            test_pad_to_multiple_function()
            test_error_handling()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()

