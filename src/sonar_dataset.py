#!/usr/bin/env python3
"""
Sonar Mine Detection Dataset Class
==================================

A comprehensive PyTorch dataset class for loading and preprocessing 
side scan sonar imagery for mine detection tasks.

Features:
- YOLO format annotation parsing
- Mixed resolution handling with smart resizing
- Sonar-specific preprocessing and augmentations  
- Temporal data splitting strategies
- Class balancing and hard negative mining
- Comprehensive validation and debugging tools
- Framework-agnostic design with PyTorch integration

Author: Sonar Mine Detection Team
Date: 2024
"""

import os
import sys
import cv2
import yaml
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from collections import defaultdict, Counter

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SonarMineDataset(Dataset):
    """
    PyTorch Dataset class for Sonar Mine Detection
    
    Handles YOLO format annotations for underwater mine detection in side scan sonar imagery.
    Supports temporal splitting, class balancing, and sonar-specific preprocessing.
    
    Args:
        data_path (str): Path to the Data directory containing year subdirectories
        years (List[str]): List of years to include in dataset (e.g., ['2010', '2015'])
        split_type (str): Type of split - 'train', 'val', or 'test'
        image_size (Tuple[int, int]): Target image size (height, width)
        augmentations (Optional[Callable]): Albumentations pipeline for data augmentation
        normalize (bool): Whether to normalize images to [0,1] range
        cache_images (bool): Whether to cache images in memory for faster loading
        balance_strategy (Optional[str]): Class balancing strategy - 'oversample_positive', 'undersample_negative', or None
        enhance_sonar (bool): Whether to apply sonar-specific image enhancements
        class_mapping (Dict[int, str]): Mapping from class IDs to class names
    """
    
    def __init__(
        self,
        data_path: str = "./Data/",
        years: List[str] = ["2010", "2015", "2017", "2018", "2021"],
        split_type: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        augmentations: Optional[Callable] = None,
        normalize: bool = True,
        cache_images: bool = False,
        balance_strategy: Optional[str] = None,
        enhance_sonar: bool = True,
        class_mapping: Dict[int, str] = {0: "MILCO", 1: "NOMBO"}
    ):
        
        self.data_path = Path(data_path)
        self.years = years
        self.split_type = split_type
        self.image_size = image_size
        self.augmentations = augmentations
        self.normalize = normalize
        self.cache_images = cache_images
        self.balance_strategy = balance_strategy
        self.enhance_sonar = enhance_sonar
        self.class_mapping = class_mapping
        
        # Initialize containers
        self.samples = []
        self.image_cache = {} if cache_images else None
        self.statistics = {}
        
        # Build dataset index
        logger.info(f"Building {split_type} dataset for years: {years}")
        self._build_dataset_index()
        
        # Apply class balancing if specified
        if balance_strategy:
            self._apply_class_balancing()
        
        # Generate dataset statistics
        self._generate_statistics()
        
        logger.info(f"Dataset created with {len(self.samples)} samples")
        
    def _build_dataset_index(self):
        """
        Build comprehensive index of all image-annotation pairs
        
        Scans the specified year directories and creates a list of samples
        containing image paths, annotation paths, and metadata.
        """
        self.samples = []
        
        for year in self.years:
            year_path = self.data_path / year
            
            if not year_path.exists():
                logger.warning(f"Year directory {year} not found at {year_path}")
                continue
                
            # Find all images for this year
            image_files = list(year_path.glob(f"*_{year}.jpg"))
            
            for img_file in image_files:
                # Corresponding annotation file
                txt_file = img_file.with_suffix('.txt')
                
                # Check if annotation exists and has content
                has_objects = (txt_file.exists() and txt_file.stat().st_size > 0)
                
                sample = {
                    'image_path': img_file,
                    'annotation_path': txt_file if txt_file.exists() else None,
                    'year': year,
                    'has_objects': has_objects,
                    'sample_id': len(self.samples),
                    'original_size': None  # Will be populated when image is loaded
                }
                
                self.samples.append(sample)
        
        logger.info(f"Found {len(self.samples)} samples across {len(self.years)} years")
        
    def _apply_class_balancing(self):
        """
        Apply class balancing strategies to handle imbalanced datasets
        
        Implements different strategies to balance positive and negative samples:
        - oversample_positive: Duplicate positive samples to match negative count
        - undersample_negative: Reduce negative samples to balance with positives
        """
        positive_samples = [s for s in self.samples if s['has_objects']]
        negative_samples = [s for s in self.samples if not s['has_objects']]
        
        logger.info(f"Before balancing: {len(positive_samples)} positive, {len(negative_samples)} negative")
        
        if self.balance_strategy == 'oversample_positive':
            if len(negative_samples) > len(positive_samples):
                # Oversample positive examples to match negative count
                target_count = len(negative_samples)
                balanced_positives = np.random.choice(
                    positive_samples, size=target_count, replace=True
                ).tolist()
                self.samples = balanced_positives + negative_samples
                
        elif self.balance_strategy == 'undersample_negative':
            if len(negative_samples) > len(positive_samples):
                # Use only subset of negative examples (2:1 ratio)
                target_count = min(len(positive_samples) * 2, len(negative_samples))
                balanced_negatives = np.random.choice(
                    negative_samples, size=target_count, replace=False
                ).tolist()
                self.samples = positive_samples + balanced_negatives
        
        # Shuffle the balanced dataset
        np.random.shuffle(self.samples)
        
        # Update sample IDs
        for i, sample in enumerate(self.samples):
            sample['sample_id'] = i
            
        new_positive = len([s for s in self.samples if s['has_objects']])
        new_negative = len([s for s in self.samples if not s['has_objects']])
        logger.info(f"After balancing: {new_positive} positive, {new_negative} negative")
        
    def _generate_statistics(self):
        """Generate comprehensive dataset statistics"""
        self.statistics = {
            'total_samples': len(self.samples),
            'positive_samples': len([s for s in self.samples if s['has_objects']]),
            'negative_samples': len([s for s in self.samples if not s['has_objects']]),
            'years': list(set([s['year'] for s in self.samples])),
            'samples_per_year': Counter([s['year'] for s in self.samples])
        }
        
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load and validate image with error handling
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            np.ndarray: Loaded image in RGB format
        """
        # Check cache first
        if self.image_cache is not None and str(image_path) in self.image_cache:
            return self.image_cache[str(image_path)].copy()
        
        try:
            # Load image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Cache if enabled
            if self.image_cache is not None:
                self.image_cache[str(image_path)] = image.copy()
                
            return image
            
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            return self._get_placeholder_image()
    
    def _get_placeholder_image(self) -> np.ndarray:
        """Create a placeholder image for corrupted files"""
        h, w = self.image_size
        placeholder = np.full((h, w, 3), 128, dtype=np.uint8)  # Gray image
        
        # Add text indicating placeholder
        cv2.putText(placeholder, "CORRUPTED", (w//4, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return placeholder
    
    def _parse_yolo_annotation(self, txt_path: Path, img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse YOLO format annotation file
        
        Args:
            txt_path (Path): Path to annotation file
            img_width (int): Original image width
            img_height (int): Original image height
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Bounding boxes and labels
        """
        boxes = []
        labels = []
        
        # Handle empty or missing annotation files
        if not txt_path or not txt_path.exists() or txt_path.stat().st_size == 0:
            return np.array([]), np.array([])
        
        try:
            with open(txt_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                        
                    parts = line.split()
                    if len(parts) < 5:
                        logger.warning(f"Invalid annotation format in {txt_path}, line {line_num}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # YOLO format uses normalized coordinates [0, 1]
                        # Convert to absolute pixel coordinates
                        x_center_abs = x_center * img_width
                        y_center_abs = y_center * img_height
                        width_abs = width * img_width
                        height_abs = height * img_height
                        
                        # Convert center format to corner format (x1, y1, x2, y2)
                        x1 = x_center_abs - width_abs / 2
                        y1 = y_center_abs - height_abs / 2
                        x2 = x_center_abs + width_abs / 2
                        y2 = y_center_abs + height_abs / 2
                        
                        # Clamp coordinates to image bounds
                        x1 = max(0, min(x1, img_width - 1))
                        y1 = max(0, min(y1, img_height - 1))
                        x2 = max(x1 + 1, min(x2, img_width))
                        y2 = max(y1 + 1, min(y2, img_height))
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing line {line_num} in {txt_path}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading annotation file {txt_path}: {e}")
            
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def _resize_image_and_boxes(self, image: np.ndarray, boxes: np.ndarray, 
                               target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize image while preserving aspect ratio and adjusting bounding boxes
        
        Uses letterboxing (padding) to maintain aspect ratio and prevent distortion.
        
        Args:
            image (np.ndarray): Input image
            boxes (np.ndarray): Bounding boxes in format [x1, y1, x2, y2]
            target_size (Tuple[int, int]): Target size (height, width)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Resized image and adjusted boxes
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor (maintain aspect ratio)
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image with gray background
        padded_image = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets (center the image)
        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        
        # Place resized image in center of padded canvas
        padded_image[dy:dy+new_h, dx:dx+new_w] = resized_image
        
        # Adjust bounding boxes if any exist
        if len(boxes) > 0:
            boxes_scaled = boxes.copy()
            boxes_scaled[:, [0, 2]] = boxes_scaled[:, [0, 2]] * scale + dx  # x coordinates
            boxes_scaled[:, [1, 3]] = boxes_scaled[:, [1, 3]] * scale + dy  # y coordinates
            
            # Ensure boxes are within image bounds
            boxes_scaled[:, [0, 2]] = np.clip(boxes_scaled[:, [0, 2]], 0, target_w)
            boxes_scaled[:, [1, 3]] = np.clip(boxes_scaled[:, [1, 3]], 0, target_h)
            
            return padded_image, boxes_scaled
        
        return padded_image, boxes
    
    def _enhance_sonar_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sonar-specific image enhancements
        
        Applies contrast enhancement and noise reduction techniques
        specifically designed for side scan sonar imagery.
        
        Args:
            image (np.ndarray): Input sonar image
            
        Returns:
            np.ndarray: Enhanced sonar image
        """
        if not self.enhance_sonar:
            return image
            
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances local contrast while preventing over-amplification
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply mild Gaussian blur to reduce sonar speckle noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        
        # Convert back to RGB format
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced_rgb
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset
        
        Args:
            idx (int): Sample index
            
        Returns:
            Dict[str, Any]: Sample containing image, boxes, labels, and metadata
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        original_h, original_w = image.shape[:2]
        
        # Parse annotations
        boxes, labels = self._parse_yolo_annotation(
            sample['annotation_path'], original_w, original_h
        )
        
        # Apply sonar-specific enhancements
        if self.enhance_sonar:
            image = self._enhance_sonar_image(image)
        
        # Resize image and adjust boxes
        image, boxes = self._resize_image_and_boxes(image, boxes, self.image_size)
        
        # Calculate areas for COCO-style formatting
        areas = []
        if len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        areas = np.array(areas, dtype=np.float32)
        
        # Create sample dictionary
        sample_dict = {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'areas': areas,
            'image_id': sample['sample_id'],
            'year': sample['year'],
            'has_objects': sample['has_objects'],
            'original_size': (original_w, original_h),
            'resized_size': self.image_size,
            'image_path': str(sample['image_path']),
            'iscrowd': np.zeros(len(labels), dtype=np.int64)  # No crowd annotations
        }
        
        # Apply augmentations if specified (training only)
        if self.augmentations and self.split_type == 'train':
            sample_dict = self._apply_augmentations(sample_dict)
        else:
            # Apply basic normalization and tensor conversion
            sample_dict = self._apply_basic_transforms(sample_dict)
        
        return sample_dict
    
    def _apply_augmentations(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply augmentations using Albumentations
        
        Args:
            sample (Dict[str, Any]): Input sample
            
        Returns:
            Dict[str, Any]: Augmented sample
        """
        image = sample['image']
        boxes = sample['boxes']
        labels = sample['labels']
        
        # Convert boxes to format expected by Albumentations (x1, y1, x2, y2)
        if len(boxes) > 0:
            try:
                # Apply augmentations
                augmented = self.augmentations(
                    image=image,
                    bboxes=boxes,
                    labels=labels
                )
                
                sample['image'] = augmented['image']
                sample['boxes'] = np.array(augmented['bboxes'], dtype=np.float32)
                sample['labels'] = np.array(augmented['labels'], dtype=np.int64)
                
                # Recalculate areas after augmentation
                if len(sample['boxes']) > 0:
                    boxes_aug = sample['boxes']
                    sample['areas'] = (boxes_aug[:, 2] - boxes_aug[:, 0]) * (boxes_aug[:, 3] - boxes_aug[:, 1])
                else:
                    sample['areas'] = np.array([], dtype=np.float32)
                    
            except Exception as e:
                logger.warning(f"Augmentation failed for sample {sample['image_id']}: {e}")
                # Fall back to basic transforms
                sample = self._apply_basic_transforms(sample)
        else:
            # No boxes, apply image-only augmentations
            sample = self._apply_basic_transforms(sample)
        
        return sample
    
    def _apply_basic_transforms(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply basic transformations (normalization and tensor conversion)
        
        Args:
            sample (Dict[str, Any]): Input sample
            
        Returns:
            Dict[str, Any]: Transformed sample
        """
        image = sample['image']
        
        # Normalize image if requested
        if self.normalize:
            image = image.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
        
        # Convert to PyTorch tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        sample['image'] = image
        sample['boxes'] = torch.from_numpy(sample['boxes']).float()
        sample['labels'] = torch.from_numpy(sample['labels']).long()
        sample['areas'] = torch.from_numpy(sample['areas']).float()
        sample['image_id'] = torch.tensor(sample['image_id'], dtype=torch.long)
        sample['iscrowd'] = torch.from_numpy(sample['iscrowd']).long()
        
        return sample
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets
        
        Returns:
            torch.Tensor: Class weights for loss function
        """
        class_counts = defaultdict(int)
        
        for sample in self.samples:
            if sample['has_objects'] and sample['annotation_path']:
                _, labels = self._parse_yolo_annotation(
                    sample['annotation_path'], 100, 100  # Dummy size, we only need labels
                )
                for label in labels:
                    class_counts[label] += 1
        
        if not class_counts:
            return torch.ones(len(self.class_mapping))
        
        total_samples = sum(class_counts.values())
        weights = []
        
        for class_id in sorted(self.class_mapping.keys()):
            if class_id in class_counts:
                weight = total_samples / (len(class_counts) * class_counts[class_id])
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def validate_dataset(self) -> Tuple[List[str], Dict[str, int]]:
        """
        Comprehensive dataset validation
        
        Returns:
            Tuple[List[str], Dict[str, int]]: Issues found and statistics
        """
        issues = []
        stats = {
            'total_samples': len(self.samples),
            'corrupted_images': 0,
            'invalid_boxes': 0,
            'missing_annotations': 0,
            'empty_images': 0
        }
        
        logger.info("Validating dataset integrity...")
        
        for idx, sample in enumerate(self.samples):
            try:
                # Test image loading
                image = self._load_image(sample['image_path'])
                h, w = image.shape[:2]
                
                if h == 0 or w == 0:
                    issues.append(f"Empty image: {sample['image_path']}")
                    stats['empty_images'] += 1
                    continue
                
                # Test annotation parsing if annotation exists
                if sample['annotation_path'] and sample['annotation_path'].exists():
                    boxes, labels = self._parse_yolo_annotation(sample['annotation_path'], w, h)
                    
                    # Check if boxes are within image bounds
                    if len(boxes) > 0:
                        invalid_boxes = (
                            (boxes[:, 0] < 0) | (boxes[:, 1] < 0) |
                            (boxes[:, 2] > w) | (boxes[:, 3] > h) |
                            (boxes[:, 0] >= boxes[:, 2]) | (boxes[:, 1] >= boxes[:, 3])
                        )
                        if invalid_boxes.any():
                            issues.append(f"Invalid boxes in {sample['annotation_path']}")
                            stats['invalid_boxes'] += 1
                            
                elif sample['has_objects']:
                    issues.append(f"Missing annotation file: {sample['annotation_path']}")
                    stats['missing_annotations'] += 1
                    
            except Exception as e:
                issues.append(f"Error processing {sample['image_path']}: {e}")
                stats['corrupted_images'] += 1
        
        logger.info(f"Validation complete. Found {len(issues)} issues.")
        return issues, stats
    
    def visualize_samples(self, num_samples: int = 4, figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize random dataset samples with annotations
        
        Args:
            num_samples (int): Number of samples to visualize
            figsize (Tuple[int, int]): Figure size for matplotlib
        """
        rows = (num_samples + 1) // 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if num_samples == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Select random samples that have objects
        positive_samples = [i for i, s in enumerate(self.samples) if s['has_objects']]
        
        if len(positive_samples) < num_samples:
            sample_indices = np.random.choice(len(self), num_samples, replace=False)
        else:
            sample_indices = np.random.choice(positive_samples, num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            if i >= len(axes):
                break
                
            sample = self[idx]
            image = sample['image']
            boxes = sample['boxes']
            labels = sample['labels']
            
            # Convert tensor to numpy if needed
            if torch.is_tensor(image):
                if image.dim() == 3 and image.shape[0] == 3:  # CHW format
                    image = image.permute(1, 2, 0)
                image = image.numpy()
                
                # Denormalize if normalized
                if self.normalize:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image = image * std + mean
                    image = np.clip(image, 0, 1)
            
            # Ensure image is in [0, 1] range for display
            if image.max() > 1.0:
                image = image / 255.0
            
            axes[i].imshow(image)
            
            # Draw bounding boxes
            if torch.is_tensor(boxes):
                boxes = boxes.numpy()
            if torch.is_tensor(labels):
                labels = labels.numpy()
                
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                width, height = x2 - x1, y2 - y1
                
                # Create rectangle patch
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[i].add_patch(rect)
                
                # Add class label
                class_name = self.class_mapping.get(int(label), f"Class {label}")
                axes[i].text(
                    x1, y1 - 5, class_name,
                    color='red', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
                )
            
            # Set title and remove axes
            sample_info = f"Sample {idx} - Year {sample['year']}"
            if sample['has_objects']:
                sample_info += f" - {len(labels)} objects"
            axes[i].set_title(sample_info, fontsize=12, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset statistics
        
        Returns:
            Dict[str, Any]: Dataset statistics
        """
        stats = self.statistics.copy()
        
        # Add class distribution
        class_counts = defaultdict(int)
        total_objects = 0
        
        for sample in self.samples:
            if sample['has_objects'] and sample['annotation_path']:
                try:
                    _, labels = self._parse_yolo_annotation(
                        sample['annotation_path'], 100, 100
                    )
                    for label in labels:
                        class_counts[label] += 1
                        total_objects += 1
                except:
                    continue
        
        stats['class_distribution'] = dict(class_counts)
        stats['total_objects'] = total_objects
        stats['positive_rate'] = stats['positive_samples'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
        
        return stats
    
    def export_annotations(self, output_path: str, format: str = 'coco'):
        """
        Export annotations in different formats
        
        Args:
            output_path (str): Output file path
            format (str): Export format - 'coco', 'yolo', or 'csv'
        """
        if format == 'coco':
            self._export_coco_format(output_path)
        elif format == 'csv':
            self._export_csv_format(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_csv_format(self, output_path: str):
        """Export annotations to CSV format"""
        annotations = []
        
        for sample in self.samples:
            base_info = {
                'image_path': str(sample['image_path']),
                'year': sample['year'],
                'has_objects': sample['has_objects']
            }
            
            if sample['has_objects'] and sample['annotation_path']:
                try:
                    # Get original image size
                    image = self._load_image(sample['image_path'])
                    h, w = image.shape[:2]
                    
                    boxes, labels = self._parse_yolo_annotation(sample['annotation_path'], w, h)
                    
                    for box, label in zip(boxes, labels):
                        annotation = base_info.copy()
                        annotation.update({
                            'class_id': label,
                            'class_name': self.class_mapping.get(label, f"Class_{label}"),
                            'x1': box[0],
                            'y1': box[1],
                            'x2': box[2],
                            'y2': box[3],
                            'width': box[2] - box[0],
                            'height': box[3] - box[1]
                        })
                        annotations.append(annotation)
                except:
                    # Add row for images with annotation errors
                    annotation = base_info.copy()
                    annotation.update({
                        'class_id': -1,
                        'class_name': 'ERROR',
                        'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0,
                        'width': 0, 'height': 0
                    })
                    annotations.append(annotation)
            else:
                # Add row for negative samples
                annotation = base_info.copy()
                annotation.update({
                    'class_id': -1,
                    'class_name': 'NEGATIVE',
                    'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0,
                    'width': 0, 'height': 0
                })
                annotations.append(annotation)
        
        df = pd.DataFrame(annotations)
        df.to_csv(output_path, index=False)
        logger.info(f"Annotations exported to {output_path}")


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
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
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


class TemporalSplitter:
    """
    Utility class for creating temporal data splits
    
    Handles splitting the dataset based on years to prevent data leakage
    and simulate real-world deployment scenarios.
    """
    
    def __init__(self, train_years: List[str], val_years: List[str], test_years: List[str]):
        """
        Initialize temporal splitter
        
        Args:
            train_years (List[str]): Years to use for training
            val_years (List[str]): Years to use for validation
            test_years (List[str]): Years to use for testing
        """
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        
        # Validate no overlap between splits
        all_years = set(train_years + val_years + test_years)
        if len(all_years) != len(train_years) + len(val_years) + len(test_years):
            raise ValueError("Overlapping years detected between splits")
    
    def create_splits(self, data_path: str, **kwargs) -> Dict[str, SonarMineDataset]:
        """
        Create temporal splits
        
        Args:
            data_path (str): Path to data directory
            **kwargs: Additional arguments to pass to SonarMineDataset
            
        Returns:
            Dict[str, SonarMineDataset]: Dictionary containing train/val/test datasets
        """
        # Get augmentations
        image_size = kwargs.get('image_size', (512, 512))
        train_augmentations = get_training_augmentations(image_size)
        val_augmentations = get_validation_transforms(image_size)
        
        splits = {
            'train': SonarMineDataset(
                data_path=data_path,
                years=self.train_years,
                split_type='train',
                augmentations=train_augmentations,
                **kwargs
            ),
            'val': SonarMineDataset(
                data_path=data_path,
                years=self.val_years,
                split_type='val',
                augmentations=val_augmentations,
                **kwargs
            ),
            'test': SonarMineDataset(
                data_path=data_path,
                years=self.test_years,
                split_type='test',
                augmentations=val_augmentations,
                **kwargs
            )
        }
        
        return splits
    
    def create_dataloaders(self, data_path: str, batch_size: int = 16, 
                          num_workers: int = 4, **kwargs) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for all splits
        
        Args:
            data_path (str): Path to data directory
            batch_size (int): Batch size for training
            num_workers (int): Number of worker processes
            **kwargs: Additional arguments for SonarMineDataset
            
        Returns:
            Dict[str, DataLoader]: Dictionary containing DataLoaders
        """
        datasets = self.create_splits(data_path, **kwargs)
        dataloaders = {}
        
        for split_name, dataset in datasets.items():
            shuffle = split_name == 'train'
            drop_last = split_name == 'train'
            
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


class SonarDatasetFactory:
    """
    Factory class for creating dataset instances with different configurations
    """
    
    @staticmethod
    def from_config(config_path: str, split_type: str = 'train') -> SonarMineDataset:
        """
        Create dataset from YAML configuration file
        
        Args:
            config_path (str): Path to YAML configuration file
            split_type (str): Type of split to create
            
        Returns:
            SonarMineDataset: Configured dataset instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_config = config['dataset']
        
        # Get split-specific configuration
        years = dataset_config['splits']['temporal'][f'{split_type}_years']
        
        # Get augmentations based on split type
        image_size = tuple(dataset_config['image_size'])
        if split_type == 'train' and dataset_config.get('augmentations', {}).get('enabled', True):
            augmentations = get_training_augmentations(image_size)
        else:
            augmentations = get_validation_transforms(image_size)
        
        return SonarMineDataset(
            data_path=dataset_config['data_path'],
            years=years,
            split_type=split_type,
            image_size=image_size,
            augmentations=augmentations,
            normalize=dataset_config.get('normalize', True),
            cache_images=dataset_config.get('cache_images', False),
            balance_strategy=dataset_config.get('balance_strategy'),
            enhance_sonar=dataset_config.get('enhance_sonar', True),
            class_mapping=dataset_config.get('class_mapping', {0: "MILCO", 1: "NOMBO"})
        )
    
    @staticmethod
    def create_full_pipeline(config_path: str, batch_size: int = 16, 
                           num_workers: int = 4) -> Tuple[Dict[str, SonarMineDataset], Dict[str, DataLoader]]:
        """
        Create complete training pipeline from configuration
        
        Args:
            config_path (str): Path to configuration file
            batch_size (int): Batch size for DataLoaders
            num_workers (int): Number of worker processes
            
        Returns:
            Tuple[Dict[str, SonarMineDataset], Dict[str, DataLoader]]: Datasets and DataLoaders
        """
        splits = ['train', 'val', 'test']
        datasets = {split: SonarDatasetFactory.from_config(config_path, split) 
                   for split in splits}
        
        dataloaders = {}
        for split_name, dataset in datasets.items():
            shuffle = split_name == 'train'
            drop_last = split_name == 'train'
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
                drop_last=drop_last,
                persistent_workers=num_workers > 0
            )
            dataloaders[split_name] = dataloader
        
        return datasets, dataloaders


def create_sample_config(output_path: str = "config/dataset_config.yaml"):
    """
    Create a sample configuration file
    
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
                'temporal': {
                    'train_years': ["2010", "2015", "2017"],
                    'val_years': ["2018"],
                    'test_years': ["2021"]
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
            
            'balance_strategy': None  # Options: "oversample_positive", "undersample_negative", None
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Sample configuration saved to {output_path}")


if __name__ == "__main__":
    """
    Example usage and testing
    """
    
    # Create sample configuration
    create_sample_config()
    
    # Example 1: Basic dataset creation
    print("Creating basic dataset...")
    dataset = SonarMineDataset(
        data_path="./Data/",
        years=["2021"],  # Start with small dataset for testing
        split_type="train",
        image_size=(512, 512),
        enhance_sonar=True
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Statistics: {dataset.get_statistics()}")
    
    # Example 2: Temporal splitting
    print("\nCreating temporal splits...")
    splitter = TemporalSplitter(
        train_years=["2010", "2015"],
        val_years=["2018"], 
        test_years=["2021"]
    )
    
    datasets = splitter.create_splits("./Data/", image_size=(512, 512))
    print(f"Train: {len(datasets['train'])} samples")
    print(f"Val: {len(datasets['val'])} samples") 
    print(f"Test: {len(datasets['test'])} samples")
    
    # Example 3: DataLoader creation
    print("\nCreating DataLoaders...")
    dataloaders = splitter.create_dataloaders(
        data_path="./Data/",
        batch_size=4,
        num_workers=0,  # Set to 0 for debugging
        image_size=(512, 512)
    )
    
    # Test loading a batch
    train_loader = dataloaders['train']
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}")
        print(f"Number of targets: {len(targets)}")
        if batch_idx == 0:  # Only test first batch
            break
    
    # Example 4: Validation
    print("\nValidating dataset...")
    issues, stats = dataset.validate_dataset()
    print(f"Validation stats: {stats}")
    if issues:
        print(f"Found {len(issues)} issues (showing first 5):")
        for issue in issues[:5]:
            print(f"  - {issue}")
    
    print("\nDataset class implementation complete!") 