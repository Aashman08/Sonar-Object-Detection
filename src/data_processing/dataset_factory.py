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
    def create_full_pipeline(config_path: str, batch_size: int = 16, 
                           num_workers: int = 4):
        """
        Create complete training pipeline from configuration using stratified splitting
        
        Args:
            config_path (str): Path to configuration file
            batch_size (int): Batch size for DataLoaders
            num_workers (int): Number of worker processes
            
        Returns:
            Tuple[Dict[str, SonarMineDataset], Dict[str, DataLoader]]: Datasets and DataLoaders
        """
        # Import here to avoid circular imports
        from .data_splitters import StratifiedSplitter
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_config = config['dataset']
        training_config = config.get('training', {})
        
        # Override batch_size and num_workers from config if provided
        batch_size = training_config.get('batch_size', batch_size)
        num_workers = training_config.get('num_workers', num_workers)
        
        # Use stratified splitting
        stratified_config = dataset_config['splits'].get('stratified', {})
        splitter = StratifiedSplitter(
            train_ratio=stratified_config.get('train_ratio', 0.7),
            val_ratio=stratified_config.get('val_ratio', 0.15),
            test_ratio=stratified_config.get('test_ratio', 0.15),
            random_state=stratified_config.get('random_state', 42),
            balance_strategy=dataset_config.get('balance_strategy', 'oversample_positive')
        )
        
        # Extract dataset arguments
        dataset_kwargs = {
            'image_size': tuple(dataset_config['image_size']),
            'normalize': dataset_config.get('normalize', True),
            'cache_images': dataset_config.get('cache_images', False),
            'enhance_sonar': dataset_config.get('enhance_sonar', True),
            'class_mapping': dataset_config.get('class_mapping', {0: "MILCO", 1: "NOMBO"})
        }
        
        datasets = splitter.create_splits(dataset_config['data_path'], **dataset_kwargs)
        dataloaders = splitter.create_dataloaders(
            datasets=datasets,  # ← FIXED! Reuse existing datasets
            batch_size=batch_size, 
            num_workers=num_workers
        )
        
        return datasets, dataloaders