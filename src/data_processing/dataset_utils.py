#!/usr/bin/env python3
"""
Dataset Utilities
=================

Utility functions for PyTorch dataset operations.
Contains collate functions and other dataset helpers.

Author: Sonar Mine Detection Team
Date: 2024
"""

import torch
from typing import Dict, List, Tuple, Any


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for DataLoader to handle variable-sized annotations
    
    Args:
        batch (List[Dict[str, Any]]): Batch of samples
        
    Returns:
        Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]: Batched images and targets
    """
    images = []
    targets = []
    
    for sample in batch:
        images.append(sample['image'])
        
        # Create target dictionary for each image
        target = {
            'boxes': sample['boxes'],
            'labels': sample['labels'],
            'image_id': sample['image_id'],
            'area': sample['areas'],
            'iscrowd': sample['iscrowd']
        }
        targets.append(target)
    
    # Stack images into a single tensor
    images = torch.stack(images, dim=0)
    
    return images, targets