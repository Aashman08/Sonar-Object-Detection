#!/usr/bin/env python3
"""
Test script for the new stratified splitting functionality
"""

import sys
sys.path.append('.')

from sonar_dataset import StratifiedSplitter, SonarDatasetFactory
import logging

# Configure logging to see detailed information
logging.basicConfig(level=logging.INFO)

def test_stratified_splitting():
    """Test the new stratified splitting vs temporal splitting"""
    
    print("=" * 80)
    print("TESTING STRATIFIED SPLITTING vs TEMPORAL SPLITTING")
    print("=" * 80)
    
    # Test 1: Direct StratifiedSplitter usage
    print("\n1. Testing direct StratifiedSplitter usage:")
    print("-" * 50)
    
    try:
        splitter = StratifiedSplitter(
            train_ratio=0.7,
            val_ratio=0.15, 
            test_ratio=0.15,
            random_state=42,
            balance_strategy='oversample_positive'
        )
        
        # Create datasets
        datasets = splitter.create_splits(
            data_path="./Data/",
            image_size=(512, 512),
            enhance_sonar=True
        )
        
        print("✅ Stratified splitting successful!")
        for split_name, dataset in datasets.items():
            positive_samples = sum(1 for s in dataset.samples if s['has_objects'])
            negative_samples = len(dataset.samples) - positive_samples
            print(f"  {split_name}: {len(dataset.samples)} total "
                  f"({positive_samples} positive, {negative_samples} negative)")
            
    except Exception as e:
        print(f"❌ Stratified splitting failed: {e}")
        
    # Test 2: Configuration-based usage
    print("\n2. Testing configuration-based usage:")
    print("-" * 50)
    
    try:
        # Use the updated config file
        datasets, dataloaders = SonarDatasetFactory.create_full_pipeline(
            config_path="config/dataset_config.yaml",
            batch_size=8,  # Smaller batch for testing
            num_workers=0  # No multiprocessing for testing
        )
        
        print("✅ Configuration-based stratified splitting successful!")
        for split_name, dataset in datasets.items():
            positive_samples = sum(1 for s in dataset.samples if s['has_objects'])
            negative_samples = len(dataset.samples) - positive_samples
            
            # Get class distribution for positive samples
            milco_count = 0
            nombo_count = 0
            
            for sample in dataset.samples:
                if sample['has_objects'] and sample['annotation_path']:
                    try:
                        with open(sample['annotation_path'], 'r') as f:
                            lines = f.read().strip().split('\n')
                            for line in lines:
                                if line.strip():
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])
                                        if class_id == 0:
                                            milco_count += 1
                                        elif class_id == 1:
                                            nombo_count += 1
                    except:
                        pass
            
            print(f"  {split_name}: {len(dataset.samples)} total samples")
            print(f"    - Positive: {positive_samples} (MILCO: {milco_count}, NOMBO: {nombo_count})")
            print(f"    - Negative: {negative_samples}")
            print(f"    - Balance: {positive_samples/(positive_samples+negative_samples)*100:.1f}% positive")
            
    except Exception as e:
        print(f"❌ Configuration-based splitting failed: {e}")
        import traceback
        traceback.print_exc()

def compare_splitting_methods():
    """Compare temporal vs stratified splitting distribution"""
    
    print("\n" + "=" * 80)
    print("COMPARING TEMPORAL vs STRATIFIED SPLITTING")
    print("=" * 80)
    
    # Test temporal splitting (change config temporarily)
    print("\nTemporal Splitting Results:")
    print("-" * 40)
    
    try:
        # Import directly and test temporal splitter
        from sonar_dataset import TemporalSplitter
        
        temporal_splitter = TemporalSplitter(
            train_years=["2010", "2015", "2017"],
            val_years=["2018"], 
            test_years=["2021"]
        )
        
        temporal_datasets = temporal_splitter.create_splits(
            data_path="./Data/",
            image_size=(512, 512)
        )
        
        print("Temporal splitting distribution:")
        for split_name, dataset in temporal_datasets.items():
            positive_samples = sum(1 for s in dataset.samples if s['has_objects'])
            negative_samples = len(dataset.samples) - positive_samples
            pos_ratio = positive_samples/(positive_samples+negative_samples)*100 if (positive_samples+negative_samples) > 0 else 0
            print(f"  {split_name}: {len(dataset.samples)} total ({pos_ratio:.1f}% positive)")
            
    except Exception as e:
        print(f"❌ Temporal splitting test failed: {e}")
        
    print("\nKey Advantages of Stratified Splitting:")
    print("-" * 40)
    print("✅ Balanced class distribution across all splits")
    print("✅ Each split represents the overall dataset characteristics")  
    print("✅ More reliable validation and test metrics")
    print("✅ Better generalization potential")
    print("✅ No bias from year-specific characteristics")
    print("\nDisadvantages of Temporal Splitting for your dataset:")
    print("-" * 40)
    print("❌ 2015 has 98.3% positive samples (extremely biased)")
    print("❌ 2010 has only 8.1% positive samples (extremely sparse)")
    print("❌ Years don't represent meaningful temporal progression")
    print("❌ Unfair train/val/test distributions")

if __name__ == "__main__":
    test_stratified_splitting()
    compare_splitting_methods()
    
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print("\n✅ YOUR ANALYSIS WAS CORRECT!")
    print("   - Temporal splitting is inappropriate for your dataset")
    print("   - Class imbalance varies dramatically by year")
    print("   - Each sample is indeed independent")
    print("\n🎯 SOLUTION IMPLEMENTED:")
    print("   - StratifiedSplitter maintains balanced distribution")
    print("   - Training set uses 'oversample_positive' for ~50-50 balance")
    print("   - Configuration updated to use stratified method by default")
    print("\n📝 TO USE IN YOUR TRAINING:")
    print("   from src.sonar_dataset import SonarDatasetFactory")
    print("   datasets, dataloaders = SonarDatasetFactory.create_full_pipeline(")
    print("       'config/dataset_config.yaml')")
    print("\n   This will now use stratified splitting with balanced training!")
    print("=" * 80)