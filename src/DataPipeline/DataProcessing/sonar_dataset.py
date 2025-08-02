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

from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SonarMineDataset(Dataset):
    """
    PyTorch Dataset class for Sonar Mine Detection
    
    Handles YOLO format annotations for underwater mine detection in side scan sonar imagery.
    Supports class balancing and sonar-specific preprocessing.
    
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
        clahe_clip_limit (float): CLAHE clip limit for contrast enhancement (default: 2.0)
        clahe_tile_grid_size (Tuple[int, int]): CLAHE tile grid size for local adaptation (default: (8, 8))
        class_mapping (Dict[int, str]): Mapping from class IDs to class names
        skip_init (bool): Skip initial dataset building (for external sample management)
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
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        class_mapping: Dict[int, str] = {0: "MILCO", 1: "NOMBO"},
        skip_init: bool = False
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
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.class_mapping = class_mapping
        
        # Initialize containers
        self.samples = []
        self.image_cache = {} if cache_images else None
        self.statistics = {}
        
        # Skip initialization if samples will be provided externally
        if not skip_init:
            # Build dataset index
            logger.info(f"Building {split_type} dataset for years: {years}")
            self._build_dataset_index()
            
            # Apply class balancing if specified
            if balance_strategy:
                self._apply_class_balancing()
            
            # Generate dataset statistics
            self._generate_statistics()
            
            logger.info(f"Dataset created with {len(self.samples)} samples")
        else:
            logger.info(f"Dataset initialized with pre-stratified samples from StratifiedSplitter - samples already selected and balanced")
        
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
        # Use configurable parameters from config file
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size)
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