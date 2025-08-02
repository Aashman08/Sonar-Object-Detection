#!/usr/bin/env python3
"""
Sonar Dataset Augmentations
===========================

Data augmentation pipelines for sonar mine detection tasks.
Contains training and validation transforms using Albumentations.

Author: Sonar Mine Detection Team
Date: 2024
"""

import cv2
from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentations(image_size: Tuple[int, int]) -> A.Compose:
    """
    Create Albumentations pipeline for training data augmentation
    
    Args:
        image_size (Tuple[int, int]): Target image size (height, width)
        
    Returns:
        A.Compose: Albumentations pipeline
    """
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.2, 
            rotate_limit=15, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=114,
            p=0.5
        ),
        
        # Photometric augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        
        # Sonar-specific augmentations
        A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3
        ),
        
        # Final normalization and tensor conversion
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_area=1,
        min_visibility=0.1
    ))


def get_validation_transforms(image_size: Tuple[int, int]) -> A.Compose:
    """
    Create simple transforms for validation/test data
    
    Args:
        image_size (Tuple[int, int]): Target image size (height, width)
        
    Returns:
        A.Compose: Validation transforms
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))


def get_augmentations_from_config(config: dict, image_size: Tuple[int, int], split_type: str) -> A.Compose:
    """
    Create augmentation pipeline from configuration parameters
    
    Args:
        config (dict): Augmentation configuration dictionary
        image_size (Tuple[int, int]): Target image size (height, width)
        split_type (str): 'train', 'val', or 'test'
        
    Returns:
        A.Compose: Configured augmentation pipeline
    """
    # For validation/test, use minimal transforms
    if split_type in ['val', 'test']:
        return get_validation_transforms(image_size)
    
    # For training, build augmentations from config
    if not config.get('enabled', False):
        return get_validation_transforms(image_size)
    
    transforms = []
    
    # Geometric augmentations
    if config.get('horizontal_flip', 0) > 0:
        transforms.append(A.HorizontalFlip(p=config['horizontal_flip']))
    
    if config.get('vertical_flip', 0) > 0:
        transforms.append(A.VerticalFlip(p=config['vertical_flip']))
    
    if config.get('rotation_limit', 0) > 0:
        transforms.append(A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=config['rotation_limit'],
            border_mode=cv2.BORDER_CONSTANT,
            value=114,
            p=0.5
        ))
    
    # Photometric augmentations
    brightness_contrast = config.get('brightness_contrast', 0)
    contrast_limit = config.get('contrast_limit', 0)
    if brightness_contrast > 0 or contrast_limit > 0:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=brightness_contrast,
            contrast_limit=contrast_limit,
            p=0.5
        ))
    
    # Noise
    if config.get('noise_probability', 0) > 0:
        transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=config['noise_probability']))
    
    # Final normalization and tensor conversion
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms, bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_area=1,
        min_visibility=0.1
    ))