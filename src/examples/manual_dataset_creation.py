#!/usr/bin/env python3
"""
Complete Manual Dataset Creation Workflow
=========================================

ONE comprehensive example showing the complete workflow for manual dataset creation.
This walks through: dataset creation → splitting → dataloaders → validation → export.

For production use, stick to the YAML configuration approach in main.py.

Author: Sonar Mine Detection Team
Date: 2024
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import (
    SonarMineDataset, 
    StratifiedSplitter, 
    get_training_augmentations,
    get_validation_transforms
)


def complete_manual_workflow():
    """
    Complete manual workflow demonstrating all dataset functionality
    """
    print("🔧 Complete Manual Dataset Creation Workflow")
    print("=" * 70)
    print("This example shows the FULL manual process from start to finish.")
    print("For production, prefer the YAML configuration in main.py.\n")
    
    # Step 1: Create custom stratified splitter
    print("📋 Step 1: Setting up custom stratified splitter...")
    splitter = StratifiedSplitter(
        train_ratio=0.7,     # 70% for training
        val_ratio=0.2,       # 20% for validation
        test_ratio=0.1,      # 10% for testing
        random_state=42,     # Reproducible results
        balance_strategy='oversample_positive'  # Handle class imbalance
    )
    print("✅ Splitter configured with 70/20/10 split and positive oversampling")
    
    # Step 2: Create datasets with custom settings
    print("\n🏗️  Step 2: Creating stratified datasets...")
    datasets = splitter.create_splits(
        data_path="./Data/", 
        image_size=(512, 512),   # High resolution for training
        enhance_sonar=True,      # Apply sonar enhancement
        normalize=True,          # Normalize images
        years=["2010", "2015", "2017", "2018", "2021"]           
    )
    
    print("✅ Datasets created:")
    print(f"   📊 Train: {len(datasets['train'])} samples")
    print(f"   📊 Val: {len(datasets['val'])} samples") 
    print(f"   📊 Test: {len(datasets['test'])} samples")
    
    # Step 3: Create DataLoaders with custom settings
    print("\n🚛 Step 3: Creating DataLoaders...")
    dataloaders = splitter.create_dataloaders(
        datasets=datasets,  # Reuse our datasets (no redundancy!)
        batch_size=8,       # Custom batch size for training
        num_workers=2,      # Parallel loading workers
    )
    print("✅ DataLoaders created with batch_size=8, num_workers=2")
    
    # Step 4: Test batch loading
    print("\n🧪 Step 4: Testing batch loading...")
    train_loader = dataloaders['train']
    try:
        batch = next(iter(train_loader))
        images, targets = batch
        print(f"✅ Batch loading successful!")
        print(f"   📐 Batch shape: {images.shape}")
        print(f"   🎯 Number of targets in batch: {len(targets)}")
        print(f"   📏 Image tensor range: [{images.min():.3f}, {images.max():.3f}]")
    except Exception as e:
        print(f"❌ Batch loading failed: {e}")
        return None
    
    # Step 5: Dataset validation
    print("\n🔍 Step 5: Validating datasets...")
    train_dataset = datasets['train']
    issues, stats = train_dataset.validate_dataset()
    
    print(f"📈 Training dataset validation results:")
    print(f"   - Total samples: {stats['total_samples']}")
    print(f"   - Corrupted images: {stats['corrupted_images']}")
    print(f"   - Invalid boxes: {stats['invalid_boxes']}")
    print(f"   - Missing annotations: {stats['missing_annotations']}")
    
    if issues:
        print(f"⚠️  Found {len(issues)} validation issues:")
        for issue in issues[:3]:  # Show first 3
            print(f"   - {issue}")
    else:
        print("✅ All validation checks passed!")
    
    # Step 6: Export annotations for analysis
    print("\n📤 Step 6: Exporting annotations...")
    try:
        train_dataset.export_annotations("manual_train_annotations.csv", format='csv')
        print("✅ Training annotations exported to: manual_train_annotations.csv")
    except Exception as e:
        print(f"⚠️  Export failed: {e}")
    
    # Step 7: Get comprehensive statistics
    print("\n📊 Step 7: Dataset statistics...")
    train_stats = train_dataset.get_statistics()
    print("📈 Training dataset statistics:")
    for key, value in train_stats.items():
        print(f"   - {key}: {value}")
    
    # Step 8: Ready for training
    print("\n" + "=" * 70)
    print("🎯 MANUAL WORKFLOW COMPLETE!")
    print("=" * 70)
    print("Your manually created datasets and dataloaders are ready:")
    print("- Training DataLoader: dataloaders['train']")
    print("- Validation DataLoader: dataloaders['val']")
    print("- Test DataLoader: dataloaders['test']")
    print("\nNext steps for training:")
    print("1. Import your model (e.g., YOLO, RetinaNet)")
    print("2. Set up training loop with dataloaders['train']")
    print("3. Validate during training with dataloaders['val']")
    print("4. Final evaluation with dataloaders['test']")
    print("\n💡 Note: For production, consider using main.py with YAML config!")
    
    return datasets, dataloaders


if __name__ == "__main__":
    datasets, dataloaders = complete_manual_workflow()