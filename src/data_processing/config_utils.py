#!/usr/bin/env python3
"""
Configuration Utilities
=======================

Utilities for creating and managing dataset configuration files.

Author: Sonar Mine Detection Team
Date: 2024
"""

import os
import yaml


def create_sample_config(output_path: str = "config/dataset_config.yaml"):
    """
    Create a sample configuration file with stratified splitting
    
    Args:
        output_path (str): Path where to save the configuration file
    """
    config = {
        'dataset': {
            'data_path': "./Data/",
            'image_size': [512, 512],
            'normalize': True,
            'cache_images': False,
            'enhance_sonar': True,
            
            'splits': {
                'stratified': {
                    'train_ratio': 0.7,
                    'val_ratio': 0.15,
                    'test_ratio': 0.15,
                    'random_state': 42
                }
            },
            
            'augmentations': {
                'enabled': True,
                'horizontal_flip': 0.5,
                'rotation_limit': 15,
                'brightness_contrast': 0.2,
                'noise_probability': 0.3
            },
            
            'class_mapping': {
                0: "MILCO",
                1: "NOMBO"
            },
            
            'balance_strategy': 'oversample_positive'  # Options: "oversample_positive", "undersample_negative", None
        },
        
        'training': {
            'batch_size': 16,
            'num_workers': 4
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Sample configuration saved to {output_path}")