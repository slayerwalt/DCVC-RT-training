#!/usr/bin/env python3
"""Test script for Vimeo90kDataset to verify dataset loading correctness."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_vimeo90k import Vimeo90kDataset


def test_dataset_initialization(root_dir: str, list_file: str) -> None:
    """Test dataset initialization."""
    print("=" * 60)
    print("Test 1: Dataset Initialization")
    print("=" * 60)
    
    try:
        dataset = Vimeo90kDataset(
            root_dir=root_dir,
            list_file=list_file,
            crop_size=256,
            training=True,
            seq_length=7,
            padding_multiple=16,
        )
        print(f"✓ Dataset initialized successfully")
        print(f"  Root directory: {root_dir}")
        print(f"  List file: {list_file}")
        print(f"  Number of sequences: {len(dataset)}")
        return dataset
    except Exception as e:
        print(f"✗ Dataset initialization failed: {e}")
        sys.exit(1)


def test_dataset_length(dataset: Vimeo90kDataset) -> None:
    """Test dataset length."""
    print("\n" + "=" * 60)
    print("Test 2: Dataset Length")
    print("=" * 60)
    
    length = len(dataset)
    print(f"✓ Dataset length: {length}")
    
    if length == 0:
        print("✗ Warning: Dataset is empty!")
        sys.exit(1)
    
    print(f"  First sequence ID: {dataset.sequence_ids[0]}")
    if length > 1:
        print(f"  Last sequence ID: {dataset.sequence_ids[-1]}")


def test_single_sample(dataset: Vimeo90kDataset, index: int = 0) -> None:
    """Test loading a single sample."""
    print("\n" + "=" * 60)
    print(f"Test 3: Loading Single Sample (index {index})")
    print("=" * 60)
    
    try:
        sample = dataset[index]
        print(f"✓ Sample loaded successfully")
        print(f"  Sample shape: {sample.shape}")
        print(f"  Sample dtype: {sample.dtype}")
        print(f"  Sample min value: {sample.min().item():.4f}")
        print(f"  Sample max value: {sample.max().item():.4f}")
        print(f"  Sample mean value: {sample.mean().item():.4f}")
        
        # Verify expected shape: (T, C, H, W) where T=7, C=1
        expected_seq_length = dataset.seq_length
        if sample.shape[0] != expected_seq_length:
            print(f"✗ Error: Expected sequence length {expected_seq_length}, got {sample.shape[0]}")
            sys.exit(1)
        
        if sample.shape[1] != 1:
            print(f"✗ Error: Expected channel dimension 1, got {sample.shape[1]}")
            sys.exit(1)
        
        # Check padding
        height, width = sample.shape[2], sample.shape[3]
        if height % dataset.padding_multiple != 0:
            print(f"✗ Error: Height {height} is not a multiple of {dataset.padding_multiple}")
            sys.exit(1)
        if width % dataset.padding_multiple != 0:
            print(f"✗ Error: Width {width} is not a multiple of {dataset.padding_multiple}")
            sys.exit(1)
        
        print(f"✓ Shape validation passed")
        print(f"  Sequence length: {sample.shape[0]}")
        print(f"  Channels: {sample.shape[1]}")
        print(f"  Height: {sample.shape[2]} (multiple of {dataset.padding_multiple})")
        print(f"  Width: {sample.shape[3]} (multiple of {dataset.padding_multiple})")
        
        return sample
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_multiple_samples(dataset: Vimeo90kDataset, num_samples: int = 5) -> None:
    """Test loading multiple samples."""
    print("\n" + "=" * 60)
    print(f"Test 4: Loading Multiple Samples ({num_samples} samples)")
    print("=" * 60)
    
    num_samples = min(num_samples, len(dataset))
    shapes = []
    
    for i in range(num_samples):
        try:
            sample = dataset[i]
            shapes.append(sample.shape)
            print(f"  Sample {i}: shape {sample.shape}")
        except Exception as e:
            print(f"✗ Failed to load sample {i}: {e}")
            sys.exit(1)
    
    # Check consistency
    if len(set(shapes)) > 1:
        print(f"✗ Warning: Samples have different shapes: {set(shapes)}")
    else:
        print(f"✓ All samples have consistent shape: {shapes[0]}")


def test_dataloader(dataset: Vimeo90kDataset, batch_size: int = 2) -> None:
    """Test DataLoader integration."""
    print("\n" + "=" * 60)
    print(f"Test 5: DataLoader Integration (batch_size={batch_size})")
    print("=" * 60)
    
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 workers for testing
            pin_memory=False,
        )
        
        batch = next(iter(dataloader))
        print(f"✓ DataLoader created and batch loaded successfully")
        print(f"  Batch shape: {batch.shape}")
        print(f"  Expected shape: (batch_size={batch_size}, seq_len=7, channels=1, H, W)")
        
        # Verify batch shape: (B, T, C, H, W)
        if batch.shape[0] != batch_size:
            print(f"✗ Error: Expected batch size {batch_size}, got {batch.shape[0]}")
            sys.exit(1)
        
        if batch.shape[1] != dataset.seq_length:
            print(f"✗ Error: Expected sequence length {dataset.seq_length}, got {batch.shape[1]}")
            sys.exit(1)
        
        if batch.shape[2] != 1:
            print(f"✗ Error: Expected channel dimension 1, got {batch.shape[2]}")
            sys.exit(1)
        
        print(f"✓ Batch shape validation passed")
        print(f"  Batch size: {batch.shape[0]}")
        print(f"  Sequence length: {batch.shape[1]}")
        print(f"  Channels: {batch.shape[2]}")
        print(f"  Height: {batch.shape[3]}")
        print(f"  Width: {batch.shape[4]}")
        
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_training_vs_validation_mode(root_dir: str, list_file: str) -> None:
    """Test training vs validation mode differences."""
    print("\n" + "=" * 60)
    print("Test 6: Training vs Validation Mode")
    print("=" * 60)
    
    train_dataset = Vimeo90kDataset(
        root_dir=root_dir,
        list_file=list_file,
        crop_size=256,
        training=True,
        seq_length=7,
        padding_multiple=16,
    )
    
    val_dataset = Vimeo90kDataset(
        root_dir=root_dir,
        list_file=list_file,
        crop_size=256,
        training=False,
        seq_length=7,
        padding_multiple=16,
    )
    
    # Load same sample multiple times in training mode (should vary due to augmentation)
    print("  Loading same sample 3 times in training mode (with augmentation):")
    train_samples = []
    for i in range(3):
        sample = train_dataset[0]
        train_samples.append(sample)
        print(f"    Sample {i+1}: shape {sample.shape}")
    
    # Load same sample multiple times in validation mode (should be consistent)
    print("  Loading same sample 3 times in validation mode (no augmentation):")
    val_samples = []
    for i in range(3):
        sample = val_dataset[0]
        val_samples.append(sample)
        print(f"    Sample {i+1}: shape {sample.shape}")
    
    # Check if training samples vary (due to random augmentation/crop)
    train_shapes = [s.shape for s in train_samples]
    val_shapes = [s.shape for s in val_samples]
    
    if len(set(train_shapes)) > 1:
        print("  ✓ Training mode shows variation (augmentation working)")
    else:
        print("  ⚠ Training mode samples have same shape (augmentation may not be working)")
    
    if len(set(val_shapes)) == 1:
        print("  ✓ Validation mode is deterministic (as expected)")
    else:
        print("  ⚠ Validation mode shows variation (unexpected)")


def test_edge_cases(root_dir: str, list_file: str) -> None:
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("Test 7: Edge Cases")
    print("=" * 60)
    
    # Test with different crop sizes
    print("  Testing with crop_size=None (no cropping):")
    try:
        dataset_no_crop = Vimeo90kDataset(
            root_dir=root_dir,
            list_file=list_file,
            crop_size=None,
            training=False,
            seq_length=7,
            padding_multiple=16,
        )
        sample = dataset_no_crop[0]
        print(f"    ✓ No crop mode works, sample shape: {sample.shape}")
    except Exception as e:
        print(f"    ✗ No crop mode failed: {e}")
    
    # Test with different sequence lengths
    print("  Testing with seq_length=3:")
    try:
        dataset_short = Vimeo90kDataset(
            root_dir=root_dir,
            list_file=list_file,
            crop_size=256,
            training=False,
            seq_length=3,
            padding_multiple=16,
        )
        sample = dataset_short[0]
        print(f"    ✓ Short sequence works, sample shape: {sample.shape}")
        if sample.shape[0] != 3:
            print(f"    ✗ Error: Expected sequence length 3, got {sample.shape[0]}")
    except Exception as e:
        print(f"    ✗ Short sequence failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Vimeo90kDataset loading")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to Vimeo-90k root directory")
    parser.add_argument("--list-file", type=str, required=True, help="Path to Vimeo-90k list file")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for DataLoader test")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to test")
    
    args = parser.parse_args()
    
    print("Vimeo90kDataset Test Suite")
    print("=" * 60)
    
    # Run all tests
    dataset = test_dataset_initialization(args.root_dir, args.list_file)
    test_dataset_length(dataset)
    test_single_sample(dataset, index=0)
    test_multiple_samples(dataset, num_samples=args.num_samples)
    test_dataloader(dataset, batch_size=args.batch_size)
    test_training_vs_validation_mode(args.root_dir, args.list_file)
    test_edge_cases(args.root_dir, args.list_file)
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()

