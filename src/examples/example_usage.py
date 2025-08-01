#!/usr/bin/env python3
"""
Example Usage of SonarMineDataset Class
=======================================

This script demonstrates various ways to use the SonarMineDataset class
for loading and preprocessing sonar mine detection data.

Examples included:
1. Basic dataset creation
2. Temporal splitting
3. DataLoader integration
4. Data augmentation
5. Visualization
6. Configuration-based setup
7. Before/After scaling and augmentation visualization
8. Sonar-specific enhancement visualization

Usage:
  python example_usage.py           # Run all examples
  python example_usage.py -viz_scale  # Show scaling visualization only
  python example_usage.py -aug     # Show augmentation visualization only
  python example_usage.py -enhance # Show sonar enhancement visualization only
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Import our custom dataset classes
from sonar_dataset import (
    SonarMineDataset,
    TemporalSplitter,
    SonarDatasetFactory,
    get_training_augmentations,
    get_validation_transforms
)


def example_1_basic_dataset():
    """Example 1: Basic dataset creation and exploration"""
    print("="*60)
    print("EXAMPLE 1: Basic Dataset Creation")
    print("="*60)
    
    # Create a basic dataset
    dataset = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],  # Start with 2021 data (small but rich)
        split_type="train",
        image_size=(512, 512),
        enhance_sonar=True,
        normalize=True
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Get and display statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Load a single sample
    sample = dataset[0]
    print(f"\nSample structure:")
    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {value}")
    
    return dataset

def example_2_dataloader_integration():
    """Example 3: DataLoader integration with batching"""
    print("\n" + "="*60)
    print("EXAMPLE 3: DataLoader Integration")
    print("="*60)
    
    # Create temporal splitter
    splitter = TemporalSplitter(
        train_years=["2015"],  # Use 2015 for fast testing (high density)
        val_years=["2021"],
        test_years=["2018"]
    )
    
    # Create DataLoaders
    dataloaders = splitter.create_dataloaders(
        data_path="./Data/",
        batch_size=4,
        num_workers=0,  # Set to 0 for compatibility
        image_size=(512, 512)  # Standard size for processing
    )
    
    print("DataLoaders created:")
    for split_name, dataloader in dataloaders.items():
        print(f"  {split_name}: {len(dataloader)} batches")
    
    # Test loading a batch
    train_loader = dataloaders['train']
    print(f"\nTesting batch loading from train loader...")
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Number of targets: {len(targets)}")
        
        # Show target structure for first image
        if len(targets) > 0:
            target = targets[0]
            print(f"  First target structure:")
            for key, value in target.items():
                if torch.is_tensor(value):
                    print(f"    {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"    {key}: {value}")
        
        if batch_idx >= 1:  # Only show first 2 batches
            break
    
    return dataloaders


def example_3_augmentation_comparison():
    """Example 4: Compare original vs augmented data"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Data Augmentation Comparison")
    print("="*60)
    
    # Create dataset without augmentation
    dataset_orig = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],
        split_type="train",
        image_size=(512, 512),
        augmentations=None,  # No augmentation
        normalize=False      # Don't normalize for visualization
    )
    
    # Create dataset with augmentation
    augmentations = get_training_augmentations((512, 512))
    dataset_aug = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],
        split_type="train",
        image_size=(512, 512),
        augmentations=augmentations,
        normalize=False      # Don't normalize for visualization
    )
    
    print(f"Datasets created:")
    print(f"  Original: {len(dataset_orig)} samples")
    print(f"  Augmented: {len(dataset_aug)} samples")
    
    # Find a sample with objects for better visualization
    sample_idx = None
    for i in range(min(20, len(dataset_orig))):
        sample = dataset_orig[i]
        if len(sample['boxes']) > 0:
            sample_idx = i
            break
    
    if sample_idx is not None:
        print(f"Using sample {sample_idx} for comparison (has {len(dataset_orig[sample_idx]['boxes'])} objects)")
        
        # Get original and augmented versions
        orig_sample = dataset_orig[sample_idx]
        aug_sample = dataset_aug[sample_idx]
        
        print(f"Original image shape: {orig_sample['image'].shape}")
        print(f"Augmented image shape: {aug_sample['image'].shape}")
        print(f"Original boxes: {len(orig_sample['boxes'])}")
        print(f"Augmented boxes: {len(aug_sample['boxes'])}")
    else:
        print("No samples with objects found in first 20 samples")
    
    return dataset_orig, dataset_aug


def example_4_visualization():
    """Example 5: Visualize dataset samples"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Dataset Visualization")
    print("="*60)
    
    # Create dataset for visualization
    dataset = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],  # Use 2021 - good mix of positive/negative
        split_type="train",
        image_size=(512, 512),
        normalize=False,  # Don't normalize for better visualization
        enhance_sonar=True
    )
    
    print(f"Visualizing samples from dataset with {len(dataset)} samples")
    
    try:
        # Visualize some samples
        dataset.visualize_samples(num_samples=4, figsize=(15, 10))
        print("Visualization complete! Check the displayed plot.")
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("This might be due to display/matplotlib issues in the environment")
    
    return dataset


def example_5_config_based_setup():
    """Example 6: Configuration-based dataset setup"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Configuration-based Setup")
    print("="*60)
    
    config_path = "config/dataset_config.yaml"
    
    # Check if config file exists
    if not Path(config_path).exists():
        print(f"Configuration file not found at {config_path}")
        print("Creating sample configuration...")
        from sonar_dataset import create_sample_config
        create_sample_config(config_path)
    
    try:
        # Create datasets from configuration
        datasets, dataloaders = SonarDatasetFactory.create_full_pipeline(
            config_path=config_path,
            batch_size=8,
            num_workers=0
        )
        
        print("Datasets created from configuration:")
        for split_name, dataset in datasets.items():
            stats = dataset.get_statistics()
            print(f"  {split_name:5}: {len(dataset):4} samples, "
                  f"{stats['positive_samples']:3} positive, "
                  f"years: {stats['years']}")
        
        print("\nDataLoaders created:")
        for split_name, dataloader in dataloaders.items():
            print(f"  {split_name}: {len(dataloader)} batches of size {dataloader.batch_size}")
        
        return datasets, dataloaders
        
    except Exception as e:
        print(f"Configuration-based setup failed: {e}")
        return None, None


def example_6_dataset_validation():
    """Example 7: Dataset validation and integrity checks"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Dataset Validation")
    print("="*60)
    
    # Create dataset for validation
    dataset = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],  # Small dataset for quick validation
        split_type="test",
        image_size=(512, 512)
    )
    
    print(f"Validating dataset with {len(dataset)} samples...")
    
    # Run validation
    issues, stats = dataset.validate_dataset()
    
    print(f"\nValidation Results:")
    print(f"Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if issues:
        print(f"\nIssues found ({len(issues)} total):")
        for i, issue in enumerate(issues[:5]):  # Show first 5 issues
            print(f"  {i+1}. {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")
    else:
        print("\nNo issues found! Dataset is valid.")
    
    return dataset, issues, stats


def example_7_export_annotations():
    """Example 8: Export annotations to different formats"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Export Annotations")
    print("="*60)
    
    # Create dataset
    dataset = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],
        split_type="test",
        image_size=(512, 512)
    )
    
    # Export to CSV format
    output_path = "exports/annotations_2021.csv"
    
    try:
        # Create export directory
        Path("exports").mkdir(exist_ok=True)
        
        # Export annotations
        dataset.export_annotations(output_path, format='csv')
        print(f"Annotations exported to {output_path}")
        
        # Show some statistics about the export
        import pandas as pd
        df = pd.read_csv(output_path)
        print(f"\nExport statistics:")
        print(f"  Total rows: {len(df)}")
        print(f"  Unique images: {df['image_path'].nunique()}")
        print(f"  Class distribution:")
        class_dist = df['class_name'].value_counts()
        for class_name, count in class_dist.items():
            print(f"    {class_name}: {count}")
        
    except Exception as e:
        print(f"Export failed: {e}")
    
    return dataset


def example_8_scaling_visualization():
    """Example 9: Visualize image before/after scaling"""
    print("="*60)
    print("EXAMPLE 9: Before/After Scaling Visualization")
    print("="*60)
    
    # Create dataset without normalization for better visualization
    dataset = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],
        split_type="train", 
        image_size=(512, 512),
        enhance_sonar=False,  # Disable enhancements for clearer comparison
        normalize=False       # Don't normalize for visualization
    )
    
    if len(dataset) == 0:
        print("No data found! Please check your data path.")
        return None
    
    # Find a sample with objects for better visualization
    sample_idx = 0
    for i in range(min(10, len(dataset))):
        if dataset.samples[i]['has_objects']:
            sample_idx = i
            break
    
    sample_info = dataset.samples[sample_idx]
    print(f"Visualizing sample: {sample_info['image_path'].name}")
    
    # Load original image
    original_image = dataset._load_image(sample_info['image_path'])
    original_h, original_w = original_image.shape[:2]
    print(f"Original size: {original_w}x{original_h}")
    
    # Parse original annotations
    boxes, labels = dataset._parse_yolo_annotation(
        sample_info['annotation_path'], original_w, original_h
    )
    
    # Apply scaling
    scaled_image, scaled_boxes = dataset._resize_image_and_boxes(
        original_image, boxes, (512, 512)
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image\\n{original_w}x{original_h}', fontsize=14)
    axes[0].axis('off')
    
    # Draw original bounding boxes
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = plt.Rectangle((x1, y1), width, height, 
                           linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, y1-5, f'Class {label}', color='red', fontsize=10, weight='bold')
    
    # Scaled image
    axes[1].imshow(scaled_image)
    axes[1].set_title(f'Scaled Image\\n512x512', fontsize=14)
    axes[1].axis('off')
    
    # Draw scaled bounding boxes
    for box, label in zip(scaled_boxes, labels):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = plt.Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x1, y1-5, f'Class {label}', color='red', fontsize=10, weight='bold')
    
    plt.suptitle('Image Scaling Comparison', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\\nScaling results:")
    print(f"  Objects found: {len(boxes)}")
    print(f"  Scale factor: {min(512/original_w, 512/original_h):.3f}")
    
    return dataset


def example_9_augmentation_visualization():
    """Example 10: Visualize image before/after augmentation"""
    print("="*60) 
    print("EXAMPLE 10: Before/After Augmentation Visualization")
    print("="*60)
    
    # Create datasets - one without augmentation, one with
    dataset_orig = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],
        split_type="train",
        image_size=(512, 512),
        augmentations=None,  # No augmentation
        normalize=False      # Don't normalize for visualization
    )
    
    augmentations = get_training_augmentations((512, 512))
    dataset_aug = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],
        split_type="train", 
        image_size=(512, 512),
        augmentations=augmentations,
        normalize=False      # Don't normalize for visualization
    )
    
    if len(dataset_orig) == 0:
        print("No data found! Please check your data path.")
        return None
    
    # Find a sample with objects
    sample_idx = 0
    for i in range(min(10, len(dataset_orig))):
        if dataset_orig.samples[i]['has_objects']:
            sample_idx = i
            break
    
    print(f"Visualizing sample {sample_idx}")
    
    # Get the same sample from both datasets
    original_sample = dataset_orig[sample_idx]
    augmented_sample = dataset_aug[sample_idx]
    
    # Create visualization comparing multiple augmented versions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Convert tensor to numpy for visualization
    orig_img = original_sample['image']
    if torch.is_tensor(orig_img):
        orig_img = orig_img.numpy()
        # Convert from (C, H, W) to (H, W, C) if needed
        if orig_img.shape[0] == 3:  # RGB channels first
            orig_img = np.transpose(orig_img, (1, 2, 0))
        orig_img = orig_img.astype(np.uint8)
    else:
        orig_img = orig_img.astype(np.uint8)
    
    # Original image (top left)
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title('Original Image', fontsize=14, weight='bold')
    axes[0, 0].axis('off')
    
    # Convert boxes and labels to numpy if they're tensors
    orig_boxes = original_sample['boxes']
    orig_labels = original_sample['labels']
    if torch.is_tensor(orig_boxes):
        orig_boxes = orig_boxes.numpy()
    if torch.is_tensor(orig_labels):
        orig_labels = orig_labels.numpy()
    
    # Draw original bounding boxes
    for box, label in zip(orig_boxes, orig_labels):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = plt.Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 0].add_patch(rect)
        axes[0, 0].text(x1, y1-5, f'Class {label}', color='red', fontsize=10, weight='bold')
    
    # Generate multiple augmented versions
    for i in range(5):
        row = i // 3 if i < 3 else 1
        col = (i % 3) if i < 3 else (i % 3) + 1
        
        if i == 0:
            continue  # Skip first position (original)
            
        # Get new augmented sample
        aug_sample = dataset_aug[sample_idx]
        
        # Convert tensor to numpy for visualization
        aug_img = aug_sample['image']
        if torch.is_tensor(aug_img):
            aug_img = aug_img.numpy()
            # Convert from (C, H, W) to (H, W, C) if needed
            if aug_img.shape[0] == 3:  # RGB channels first
                aug_img = np.transpose(aug_img, (1, 2, 0))
            aug_img = aug_img.astype(np.uint8)
        else:
            aug_img = aug_img.astype(np.uint8)
        
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(f'Augmented Version {i}', fontsize=14)
        axes[row, col].axis('off')
        
        # Convert boxes and labels to numpy if they're tensors
        aug_boxes = aug_sample['boxes']
        aug_labels = aug_sample['labels']
        if torch.is_tensor(aug_boxes):
            aug_boxes = aug_boxes.numpy()
        if torch.is_tensor(aug_labels):
            aug_labels = aug_labels.numpy()
        
        # Draw augmented bounding boxes
        for box, label in zip(aug_boxes, aug_labels):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            rect = plt.Rectangle((x1, y1), width, height,
                               linewidth=2, edgecolor='lime', facecolor='none')
            axes[row, col].add_patch(rect)
            axes[row, col].text(x1, y1-5, f'Class {label}', color='lime', fontsize=10, weight='bold')
    
    plt.suptitle('Data Augmentation Comparison\\nRed: Original, Green: Augmented', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()
    
    # Convert final counts to integers if they're tensors
    orig_count = len(original_sample['boxes'])
    aug_count = len(augmented_sample['boxes'])
    if torch.is_tensor(original_sample['boxes']):
        orig_count = original_sample['boxes'].shape[0]
    if torch.is_tensor(augmented_sample['boxes']):
        aug_count = augmented_sample['boxes'].shape[0]
    
    print(f"\\nAugmentation results:")
    print(f"  Original objects: {orig_count}")
    print(f"  Augmented objects: {aug_count}")
    print("  Note: Augmentation may vary each time due to randomness")
    
    return dataset_orig, dataset_aug


def example_10_sonar_enhancement_visualization():
    """Example 11: Visualize sonar-specific image enhancements"""
    print("="*60)
    print("EXAMPLE 11: Sonar Enhancement Visualization")
    print("="*60)
    
    # Create dataset with enhancements disabled first to get original images
    dataset_no_enhance = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],
        split_type="train", 
        image_size=(512, 512),
        enhance_sonar=False,  # Disable enhancements
        normalize=False       # Don't normalize for visualization
    )
    
    # Create dataset with enhancements enabled
    dataset_enhance = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],
        split_type="train", 
        image_size=(512, 512),
        enhance_sonar=True,   # Enable enhancements
        normalize=False       # Don't normalize for visualization
    )
    
    if len(dataset_no_enhance) == 0:
        print("No data found! Please check your data path.")
        return None
    
    # Find a sample with objects for better visualization
    sample_idx = 0
    for i in range(min(10, len(dataset_no_enhance))):
        if dataset_no_enhance.samples[i]['has_objects']:
            sample_idx = i
            break
    
    sample_info = dataset_no_enhance.samples[sample_idx]
    print(f"Visualizing sample: {sample_info['image_path'].name}")
    
    # Get raw image without any processing
    raw_image = dataset_no_enhance._load_image(sample_info['image_path'])
    original_h, original_w = raw_image.shape[:2]
    
    # Parse annotations for the raw image
    boxes, labels = dataset_no_enhance._parse_yolo_annotation(
        sample_info['annotation_path'], original_w, original_h
    )
    
    # Apply sonar enhancements manually to show the effect
    enhanced_image = dataset_enhance._enhance_sonar_image(raw_image.copy())
    
    # Scale both images to target size for fair comparison
    scaled_raw, scaled_boxes = dataset_no_enhance._resize_image_and_boxes(
        raw_image, boxes, (512, 512)
    )
    scaled_enhanced, _ = dataset_enhance._resize_image_and_boxes(
        enhanced_image, boxes, (512, 512)
    )
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Raw original image (top left)
    axes[0, 0].imshow(raw_image)
    axes[0, 0].set_title(f'Original Image\\n{original_w}x{original_h}\\n(No Enhancements)', fontsize=14)
    axes[0, 0].axis('off')
    
    # Draw original bounding boxes
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = plt.Rectangle((x1, y1), width, height, 
                           linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 0].add_patch(rect)
        axes[0, 0].text(x1, y1-5, f'Class {label}', color='red', fontsize=10, weight='bold')
    
    # Enhanced original image (top right)
    axes[0, 1].imshow(enhanced_image)
    axes[0, 1].set_title(f'Enhanced Image\\n{original_w}x{original_h}\\n(CLAHE + Noise Reduction)', fontsize=14)
    axes[0, 1].axis('off')
    
    # Draw enhanced bounding boxes
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = plt.Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor='lime', facecolor='none')
        axes[0, 1].add_patch(rect)
        axes[0, 1].text(x1, y1-5, f'Class {label}', color='lime', fontsize=10, weight='bold')
    
    # Scaled raw image (bottom left)
    axes[1, 0].imshow(scaled_raw)
    axes[1, 0].set_title('Scaled Original\\n512x512\\n(No Enhancements)', fontsize=14)
    axes[1, 0].axis('off')
    
    # Draw scaled bounding boxes
    for box, label in zip(scaled_boxes, labels):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = plt.Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor='red', facecolor='none')
        axes[1, 0].add_patch(rect)
        axes[1, 0].text(x1, y1-5, f'Class {label}', color='red', fontsize=10, weight='bold')
    
    # Scaled enhanced image (bottom right)
    axes[1, 1].imshow(scaled_enhanced)
    axes[1, 1].set_title('Scaled Enhanced\\n512x512\\n(CLAHE + Noise Reduction)', fontsize=14)
    axes[1, 1].axis('off')
    
    # Draw scaled enhanced bounding boxes
    for box, label in zip(scaled_boxes, labels):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = plt.Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor='lime', facecolor='none')
        axes[1, 1].add_patch(rect)
        axes[1, 1].text(x1, y1-5, f'Class {label}', color='lime', fontsize=10, weight='bold')
    
    plt.suptitle('Sonar Image Enhancement Comparison\\nRed: Original, Green: Enhanced', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\\nSonar Enhancement Details:")
    print(f"  Original size: {original_w}x{original_h}")
    print(f"  Objects found: {len(boxes)}")
    print(f"  Enhancements applied:")
    print(f"    • CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    print(f"    • Gaussian blur for noise reduction (3x3 kernel)")
    print(f"    • RGB conversion for model compatibility")
    print(f"  Note: Enhanced images often show better contrast and reduced noise")
    
    return dataset_no_enhance, dataset_enhance


def main():
    """Run examples based on command line arguments"""
    parser = argparse.ArgumentParser(description='Sonar Mine Detection Dataset Examples')
    parser.add_argument('-viz_scale', '--visualize-scaling', action='store_true',
                       help='Show before/after scaling visualization only')
    parser.add_argument('-aug', '--augmentation', action='store_true', 
                       help='Show before/after augmentation visualization only')
    parser.add_argument('-enhance', '--sonar-enhancement', action='store_true',
                       help='Show before/after sonar enhancement visualization only') 
    parser.add_argument('--all', action='store_true',
                       help='Run all examples (default behavior)')
    
    args = parser.parse_args()
    
    print("SONAR MINE DETECTION DATASET - EXAMPLE USAGE")
    print("=" * 80)
    
    try:
        # Run specific examples based on arguments
        if args.visualize_scaling:
            print("Running scaling visualization only...")
            dataset = example_8_scaling_visualization()
            
        elif args.augmentation:
            print("Running augmentation visualization only...")
            dataset_orig, dataset_aug = example_9_augmentation_visualization()
            
        elif args.sonar_enhancement:
            print("Running sonar enhancement visualization only...")
            dataset_no_enhance, dataset_enhance = example_10_sonar_enhancement_visualization()
            
        else:
            # Run all examples (default behavior)
            print("Running all examples...")
            dataset1 = example_1_basic_dataset()
            dataloaders3 = example_2_dataloader_integration()
            dataset_orig, dataset_aug = example_3_augmentation_comparison()
            dataset5 = example_4_visualization()
            datasets6, dataloaders6 = example_5_config_based_setup()
            dataset7, issues, stats = example_6_dataset_validation()
            dataset8 = example_7_export_annotations()
            
            print("\n" + "="*60)
            print("BONUS: Advanced Visualizations")
            print("="*60)
            dataset9 = example_8_scaling_visualization()
            dataset10_orig, dataset10_aug = example_9_augmentation_visualization()
            dataset11_no_enhance, dataset11_enhance = example_10_sonar_enhancement_visualization()
            
            print("\n" + "="*80)
            print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            print("\nNext steps:")
            print("1. Check the visualizations if they were displayed")
            print("2. Review the exported annotations in exports/")
            print("3. Modify config/dataset_config.yaml for your specific needs")
            print("4. Use the datasets with your training pipeline")
            
            print("\n" + "="*60)
            print("TIP: Run specific visualizations with:")
            print("  python example_usage.py -viz_scale")
            print("  python example_usage.py -aug")
            print("  python example_usage.py -enhance")
            print("="*60)
        
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        print("Please check that your data is in the correct format and location")
        print("\nAvailable options:")
        print("  python example_usage.py           # Run all examples")
        print("  python example_usage.py -viz_scale  # Show scaling visualization")
        print("  python example_usage.py -aug     # Show augmentation visualization")
        print("  python example_usage.py -enhance # Show sonar enhancement visualization")


if __name__ == "__main__":
    main() 