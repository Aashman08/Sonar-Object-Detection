#!/usr/bin/env python3
"""
Quick Test Script for SonarMineDataset
======================================

A simple test to verify the dataset class works correctly with your data.
Run this before using the full example_usage.py script.
"""

import sys
import torch
from pathlib import Path

# Add the src directory to Python path
sys.path.append('.')

from sonar_dataset import SonarMineDataset, TemporalSplitter


def test_basic_functionality():
    """Test basic dataset functionality"""
    print("Testing basic dataset functionality...")
    
    try:
        # Create a simple dataset with 2021 data (smallest dataset)
        dataset = SonarMineDataset(
            data_path="./Data/",
            years=["2021"],
            split_type="test",
            image_size=(512, 512),
            normalize=True,
            cache_images=False
        )
        
        print(f"✓ Dataset created successfully with {len(dataset)} samples")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Successfully loaded sample 0")
            print(f"  - Image shape: {sample['image'].shape}")
            print(f"  - Number of boxes: {len(sample['boxes'])}")
            print(f"  - Labels: {sample['labels']}")
            print(f"  - Year: {sample['year']}")
        else:
            print("✗ Dataset is empty!")
            return False
        
        # Test statistics
        stats = dataset.get_statistics()
        print(f"✓ Dataset statistics:")
        print(f"  - Total samples: {stats['total_samples']}")
        print(f"  - Positive samples: {stats['positive_samples']}")
        print(f"  - Positive rate: {stats['positive_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_dataloader():
    """Test DataLoader integration"""
    print("\nTesting DataLoader integration...")
    
    try:
        # Create a small dataset
        dataset = SonarMineDataset(
            data_path="./Data/",
            years=["2021"],
            split_type="test",
            image_size=(512, 512)
        )
        
        from torch.utils.data import DataLoader
        from sonar_dataset import collate_fn
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # Test loading one batch
        for images, targets in dataloader:
            print(f"✓ DataLoader working:")
            print(f"  - Batch size: {images.shape[0]}")
            print(f"  - Image shape: {images.shape}")
            print(f"  - Number of targets: {len(targets)}")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ Error in DataLoader: {e}")
        return False


def test_data_path():
    """Test if the data path is correct"""
    print("Testing data path...")
    
    data_path = Path("./Data/")
    
    if not data_path.exists():
        print(f"✗ Data directory not found at {data_path}")
        return False
    
    # Check for year directories
    years = ["2010", "2015", "2017", "2018", "2021"]
    found_years = []
    
    for year in years:
        year_path = data_path / year
        if year_path.exists():
            # Count files in this year
            jpg_files = list(year_path.glob(f"*_{year}.jpg"))
            txt_files = list(year_path.glob(f"*_{year}.txt"))
            found_years.append((year, len(jpg_files), len(txt_files)))
            print(f"✓ Year {year}: {len(jpg_files)} images, {len(txt_files)} annotations")
        else:
            print(f"✗ Year {year} directory not found")
    
    if not found_years:
        print("✗ No valid year directories found!")
        return False
    
    print(f"✓ Found {len(found_years)} year directories")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("SONAR MINE DETECTION DATASET - QUICK TESTS")
    print("=" * 60)
    
    tests = [
        ("Data Path", test_data_path),
        ("Basic Functionality", test_basic_functionality),
        ("DataLoader Integration", test_dataloader),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:25}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your dataset is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python example_usage.py' for comprehensive examples")
        print("2. Start building your training pipeline")
        print("3. Experiment with different configurations")
    else:
        print("\n❌ Some tests failed. Please check:")
        print("1. Data directory structure is correct")
        print("2. Required dependencies are installed")
        print("3. File permissions are set correctly")


if __name__ == "__main__":
    main() 