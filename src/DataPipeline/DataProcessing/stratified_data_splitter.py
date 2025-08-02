#!/usr/bin/env python3
"""
Data Splitting Utilities
========================

Utilities for creating stratified data splits that maintain class balance.
Contains splitters for machine learning training workflows.

Author: Sonar Mine Detection Team
Date: 2024
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

# Configure logging
logger = logging.getLogger(__name__)


class StratifiedSplitter:
    """
    Utility class for creating stratified data splits that maintain class balance
    
    Unlike temporal splitting, this ensures fair distribution of classes across
    train/validation/test splits, which is more appropriate when temporal 
    relationships don't matter and each sample is independent.
    """
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                 test_ratio: float = 0.15, random_state: int = 42,
                 balance_strategy: str = 'oversample_positive'):
        """
        Initialize stratified splitter
        
        Args:
            train_ratio (float): Proportion of data for training (default: 0.7)
            val_ratio (float): Proportion of data for validation (default: 0.15)
            test_ratio (float): Proportion of data for testing (default: 0.15)
            random_state (int): Random seed for reproducibility
            balance_strategy (str): How to balance classes in training set
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
            
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.balance_strategy = balance_strategy
        
    def _collect_all_samples(self, data_path: str, years: List[str] = None) -> List[Dict[str, Any]]:
        """
        Collect all samples from specified years with metadata for stratification
        
        Args:
            data_path (str): Path to data directory
            years (List[str]): List of years to scan, defaults to all available years
        
        Returns:
            List[Dict]: All samples with stratification keys
        """
        data_path = Path(data_path)
        all_samples = []
        
        # Use provided years or default to all available years
        if years is None:
            years = ["2010", "2015", "2017", "2018", "2021"]
        
        logger.info(f"Collecting samples from years: {years}")
        
        for year in years:
            year_path = data_path / year
            if not year_path.exists():
                logger.warning(f"Year directory {year_path} not found, skipping")
                continue
                
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(year_path.glob(f'*{ext}'))
                image_files.extend(year_path.glob(f'*{ext.upper()}'))
            
            for img_path in image_files:
                # Corresponding annotation file
                ann_path = img_path.with_suffix('.txt')
                
                # Check what type of objects are in the image
                has_objects = False
                class_labels = []
                if ann_path.exists():
                    try:
                        with open(ann_path, 'r') as f:
                            lines = f.read().strip().split('\n')
                            for line in lines:
                                if line.strip():
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        has_objects = True
                                        class_labels.append(int(parts[0]))
                    except Exception as e:
                        logger.warning(f"Error reading annotation {ann_path}: {e}")
                
                # Create meaningful stratification for mine detection
                if not has_objects:
                    strat_key = "no_object"
                else:
                    # Determine dominant mine type
                    class_counts = Counter(class_labels)
                    dominant_class = class_counts.most_common(1)[0][0]
                    
                    # Based on your class mapping: 0=MILCO (mine-like), 1=NOMBO (non-mine-like)
                    if dominant_class == 0:
                        strat_key = "mine_like"      # MILCO mines
                    elif dominant_class == 1:
                        strat_key = "non_mine_like"  # NOMBO mines  
                    else:
                        strat_key = "mine_like"      # Default to mine-like for unknown classes
                
                sample = {
                    'image_path': str(img_path),
                    'annotation_path': str(ann_path) if ann_path.exists() else None,
                    'year': year,
                    'has_objects': has_objects,
                    'class_labels': class_labels,
                    'strat_key': strat_key
                }
                all_samples.append(sample)
        
        logger.info(f"Collected {len(all_samples)} total samples")
        
        # Print initial distribution
        strat_counts = Counter([s['strat_key'] for s in all_samples])
        logger.info("Initial stratification distribution:")
        for key, count in strat_counts.items():
            logger.info(f"  {key}: {count} samples ({100*count/len(all_samples):.1f}%)")
        
        # Remove half of no-object images to balance dataset
        no_object_samples = [s for s in all_samples if s['strat_key'] == 'no_object']
        mine_like_samples = [s for s in all_samples if s['strat_key'] == 'mine_like']
        non_mine_like_samples = [s for s in all_samples if s['strat_key'] == 'non_mine_like']
        
        # Keep only half of no-object images (randomly selected)
        import random
        random.seed(self.random_state)  # Use same random state for reproducibility
        reduced_no_object = random.sample(no_object_samples, len(no_object_samples) // 2)
        
        # Combine balanced samples
        balanced_samples = mine_like_samples + non_mine_like_samples + reduced_no_object
        
        logger.info(f"Balanced dataset: {len(balanced_samples)} samples (removed {len(no_object_samples) - len(reduced_no_object)} no-object images)")
        
        # Print final distribution
        final_strat_counts = Counter([s['strat_key'] for s in balanced_samples])
        logger.info("Final balanced stratification distribution:")
        for key, count in final_strat_counts.items():
            logger.info(f"  {key}: {count} samples ({100*count/len(balanced_samples):.1f}%)")
            
        return balanced_samples
    
    def _create_stratified_splits(self, samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create stratified splits maintaining class balance
        
        Args:
            samples: All samples with stratification metadata
            
        Returns:
            Dict containing train/val/test sample lists
        """
        # Convert to arrays for sklearn
        X = np.arange(len(samples))
        y = [sample['strat_key'] for sample in samples]
        
        # First split: separate training from temp (val+test)
        temp_ratio = self.val_ratio + self.test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_ratio, 
            random_state=self.random_state, 
            stratify=y
        )
        
        # Second split: separate validation from test
        test_ratio_adjusted = self.test_ratio / temp_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        # Convert back to sample dictionaries
        splits = {
            'train': [samples[i] for i in X_train],
            'val': [samples[i] for i in X_val],
            'test': [samples[i] for i in X_test]
        }
        
        # Log split statistics
        for split_name, split_samples in splits.items():
            split_strat_counts = Counter([s['strat_key'] for s in split_samples])
            logger.info(f"{split_name.capitalize()} split ({len(split_samples)} samples):")
            for key, count in split_strat_counts.items():
                logger.info(f"  {key}: {count} samples ({100*count/len(split_samples):.1f}%)")
        
        return splits
    
    def create_splits(self, data_path: str, **kwargs):
        """
        Create stratified splits with balanced class distribution
        
        Args:
            data_path (str): Path to data directory
            **kwargs: Additional arguments to pass to SonarMineDataset
            
        Returns:
            Dict[str, SonarMineDataset]: Dictionary containing train/val/test datasets
        """
        # Import locally since sonar_dataset is now in the same package
        from .sonar_dataset import SonarMineDataset
        from .sonar_augmentations import get_augmentations_from_config
        
        # Collect all samples and create stratified splits
        years_to_use = kwargs.get('years', None)  # Extract years from kwargs
        all_samples = self._collect_all_samples(data_path, years_to_use)
        sample_splits = self._create_stratified_splits(all_samples)
        
        # Get augmentation configurations
        image_size = kwargs.get('image_size', (512, 512))
        augmentations_config = kwargs.get('augmentations_config', {})
        
        # Create datasets with balanced training
        datasets = {}
        
        for split_name, split_samples in sample_splits.items():
            # Determine augmentations and balance strategy
            augmentations = get_augmentations_from_config(
                augmentations_config, image_size, split_name
            )
            if split_name == 'train':
                # Use config value if provided, otherwise fall back to self.balance_strategy
                balance_strategy = kwargs.get('balance_strategy', self.balance_strategy)
            else:
                balance_strategy = None  # Don't balance val/test sets
            
            # Convert our sample format to dataset's expected format
            dataset_samples = []
            years_in_split = set()
            for i, sample in enumerate(split_samples):
                dataset_sample = {
                    'sample_id': i,
                    'image_path': Path(sample['image_path']),
                    'annotation_path': Path(sample['annotation_path']) if sample['annotation_path'] else None,
                    'year': sample['year'],
                    'has_objects': sample['has_objects']
                }
                dataset_samples.append(dataset_sample)
                years_in_split.add(sample['year'])
            
            # Create dataset directly with the stratified samples
            dataset = SonarMineDataset(
                data_path=data_path,
                years=list(years_in_split),
                split_type=split_name,
                augmentations=augmentations,
                balance_strategy=balance_strategy,
                image_size=kwargs.get('image_size', (512, 512)),
                normalize=kwargs.get('normalize', True),
                cache_images=kwargs.get('cache_images', False),
                enhance_sonar=kwargs.get('enhance_sonar', True),
                class_mapping=kwargs.get('class_mapping', {0: "MILCO", 1: "NOMBO"}),
                clahe_clip_limit=kwargs.get('clahe_clip_limit', 2.0),
                clahe_tile_grid_size=kwargs.get('clahe_tile_grid_size', (8, 8)),
                skip_init=True  # Skip initialization since we'll set samples manually
            )
            
            # Set the pre-stratified samples
            dataset.samples = dataset_samples
            
            # Apply class balancing only to training set
            if split_name == 'train' and balance_strategy:
                dataset._apply_class_balancing()
            
            # Generate final statistics
            dataset._generate_statistics()
            
            datasets[split_name] = dataset
            logger.info(f"Created {split_name} dataset with {len(dataset.samples)} samples")
        
        return datasets
    
    def create_dataloaders(self, data_path: str = None, batch_size: int = 16, 
                          num_workers: int = 4, datasets: Dict = None, **kwargs):
        """
        Create DataLoaders with stratified splits
        
        Args:
            data_path (str): Path to data directory (only if datasets not provided)
            batch_size (int): Batch size for DataLoaders
            num_workers (int): Number of worker processes
            datasets (Dict): Pre-created datasets to avoid redundant work
            **kwargs: Additional arguments for SonarMineDataset
            
        Returns:
            Dict[str, DataLoader]: DataLoaders for train/val/test splits
        """
        # Import here to avoid circular imports
        from .dataset_utils import collate_fn
        
        # Use provided datasets or create new ones
        if datasets is None:
            if data_path is None:
                raise ValueError("Either datasets or data_path must be provided")
            datasets = self.create_splits(data_path, **kwargs)
        else:
            print("ℹ️  Reusing existing datasets (no redundant splitting)")
        
        dataloaders = {}
        for split_name, dataset in datasets.items():
            # Different settings for different splits
            shuffle = (split_name == 'train')
            drop_last = (split_name == 'train')
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
                drop_last=drop_last,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None
            )
            dataloaders[split_name] = dataloader
        
        return dataloaders