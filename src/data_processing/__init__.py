"""
Data Processing Module
=====================

This module contains all data processing related functionality for the Sonar Mine Detection project.

Components:
- augmentations: Data augmentation pipelines for training and validation
- data_splitters: Stratified data splitting utilities
- dataset_utils: Utility functions for PyTorch dataset operations
- dataset_factory: Factory patterns for creating complete training pipelines
- config_utils: Configuration management utilities

Author: Sonar Mine Detection Team
Date: 2024
"""

from .sonar_dataset import SonarMineDataset
from .augmentations import get_training_augmentations, get_validation_transforms
from .data_splitters import StratifiedSplitter
from .dataset_utils import collate_fn
from .dataset_factory import SonarDatasetFactory
from .config_utils import create_sample_config

__all__ = [
    'SonarMineDataset',
    'get_training_augmentations',
    'get_validation_transforms', 
    'StratifiedSplitter',
    'collate_fn',
    'SonarDatasetFactory',
    'create_sample_config'
]