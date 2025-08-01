#!/usr/bin/env python3
"""
Sonar Mine Detection Dataset - Main Training Script
==================================================

Production-ready entry point for creating datasets using YAML configuration.
This is the ONLY recommended way to create datasets for training.

For manual/custom dataset creation examples, see examples/ folder.

Author: Sonar Mine Detection Team
Date: 2024
"""

from data_processing import SonarDatasetFactory, create_sample_config


def main():
    """
    Main production workflow - YAML configuration based dataset creation
    """
    print("🎯 Sonar Mine Detection Dataset Pipeline")
    print("=" * 50)
    
    # Step 1: Ensure configuration exists
    print("📄 Creating/updating configuration file...")
    create_sample_config()
    print("✅ Configuration ready at: config/dataset_config.yaml")
    
    # Step 2: Create complete pipeline from configuration (SINGLE source of truth)
    print("\n🏭 Creating complete training pipeline from configuration...")
    try:
        datasets, dataloaders = SonarDatasetFactory.create_full_pipeline(
            config_path="config/dataset_config.yaml"
        )
        
        print("✅ Pipeline created successfully!")
        print(f"   📊 Train: {len(datasets['train'])} samples")
        print(f"   📊 Val: {len(datasets['val'])} samples") 
        print(f"   📊 Test: {len(datasets['test'])} samples")
        
    except FileNotFoundError:
        print("❌ Configuration file not found!")
        print("   Run create_sample_config() to generate it.")
        return None, None
    except Exception as e:
        print(f"❌ Error creating pipeline: {e}")
        return None, None
    
    # Step 3: Quick validation test
    print("\n🧪 Running quick validation...")
    sample_dataset = datasets['train']
    issues, stats = sample_dataset.validate_dataset()
    
    if issues:
        print(f"⚠️  Found {len(issues)} validation issues:")
        for issue in issues[:3]:  # Show first 3
            print(f"   - {issue}")
    else:
        print("✅ Dataset validation passed!")
    
    print(f"📈 Dataset statistics: {stats}")
    
    # Step 4: Test batch loading
    print("\n🚛 Testing batch loading...")
    train_loader = dataloaders['train']
    try:
        batch = next(iter(train_loader))
        images, targets = batch
        print(f"✅ Batch loading successful!")
        print(f"   📐 Batch shape: {images.shape}")
        print(f"   🎯 Number of targets: {len(targets)}")
    except Exception as e:
        print(f"❌ Batch loading failed: {e}")
        return datasets, dataloaders
    
    # Step 5: Ready for training
    print("\n🎯 READY FOR TRAINING!")
    print("=" * 50)
    print("Your datasets and dataloaders are ready to use:")
    print("- Training DataLoader: dataloaders['train']")
    print("- Validation DataLoader: dataloaders['val']") 
    print("- Test DataLoader: dataloaders['test']")
    print("\nNext steps:")
    print("1. Import your model")
    print("2. Set up training loop with dataloaders['train']")
    print("3. Validate with dataloaders['val']")
    print("4. Test with dataloaders['test']")
    
    return datasets, dataloaders


if __name__ == "__main__":
    datasets, dataloaders = main()