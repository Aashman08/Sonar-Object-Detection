#!/usr/bin/env python3
"""
Dataset Factory
===============

Factory class for creating stratified dataset splits and training pipelines from configuration.

Author: Sonar Mine Detection Team
Date: 2024
"""

import yaml
from typing import Dict, Tuple
from torch.utils.data import DataLoader


class SonarDatasetFactory:
    """
    Factory class for creating stratified dataset splits and training pipelines from configuration
    """
    
    @staticmethod
    def create_full_pipeline(config_path: str, batch_size: int = None, num_workers: int = None):
        """
        Create complete training pipeline from YAML configuration
        
        Args:
            config_path (str): Path to configuration file
            batch_size (int, optional): Override batch size from config
            num_workers (int, optional): Override num_workers from config
            
        Returns:
            Tuple[Dict[str, SonarMineDataset], Dict[str, DataLoader]]: Datasets and DataLoaders
        """
        from .stratified_data_splitter import StratifiedSplitter
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_config = config['dataset']
        training_config = config.get('training', {})
        
        # Get training parameters (use override if provided, otherwise config, otherwise defaults)
        batch_size = batch_size or training_config.get('batch_size', 16)
        num_workers = num_workers or training_config.get('num_workers', 4)
        
        # Create stratified splitter from config
        splits_config = dataset_config.get('splits', {}).get('stratified', {})
        splitter = StratifiedSplitter(
            train_ratio=splits_config.get('train_ratio', 0.7),
            val_ratio=splits_config.get('val_ratio', 0.15),
            test_ratio=splits_config.get('test_ratio', 0.15),
            random_state=splits_config.get('random_state', 42),
            balance_strategy=dataset_config.get('balance_strategy', 'oversample_positive')
        )
        
        # Pass all dataset config parameters to splitter
        dataset_kwargs = dataset_config.copy()
        # Remove parameters that splitter doesn't need
        dataset_kwargs.pop('splits', None)
        dataset_kwargs.pop('data_path', None)  # Remove data_path since it's passed as positional arg
        # Keep balance_strategy in kwargs so splitter can use config value over self value
        
        # Rename augmentations to augmentations_config for splitter compatibility
        if 'augmentations' in dataset_kwargs:
            dataset_kwargs['augmentations_config'] = dataset_kwargs.pop('augmentations')
        
        # Convert image_size to tuple
        if 'image_size' in dataset_kwargs:
            dataset_kwargs['image_size'] = tuple(dataset_kwargs['image_size'])
            
        # Flatten CLAHE settings to individual parameters
        if 'clahe_settings' in dataset_kwargs:
            clahe_settings = dataset_kwargs.pop('clahe_settings')
            dataset_kwargs['clahe_clip_limit'] = clahe_settings.get('clip_limit', 2.0)
            dataset_kwargs['clahe_tile_grid_size'] = tuple(clahe_settings.get('tile_grid_size', [8, 8]))
        
        # Create datasets and dataloaders
        datasets = splitter.create_splits(dataset_config['data_path'], **dataset_kwargs)
        dataloaders = splitter.create_dataloaders(
            datasets=datasets,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        return datasets, dataloaders