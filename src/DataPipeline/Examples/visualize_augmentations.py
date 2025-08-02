#!/usr/bin/env python3
"""
Data Augmentation Visualization Tool
====================================

Intelligent tool for comparing different augmentation parameters on sonar images.
Helps you find optimal augmentation settings for your mine detection training.

Usage:
    python visualize_augmentations.py                         # Use default config
    python visualize_augmentations.py --config custom.yaml    # Use custom config
    python visualize_augmentations.py --simulate-training     # Show training batch simulation
    
Author: Sonar Mine Detection Team
Date: 2024
"""

import sys
import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random
import albumentations as A

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataProcessing import SonarMineDataset


class AugmentationOptimizer:
    """Smart augmentation parameter optimization for sonar images"""
    
    def __init__(self, config_path: str = "YAML_Config/augmentation_config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directory
        output_dir = Path(self.config['visualization']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        self.output_dir = output_dir
        
    def find_sample_images(self) -> List[Dict[str, Any]]:
        """Find random images with objects for testing"""
        config = self.config['image_selection']
        
        # Create temporary dataset to find images with objects
        dataset = SonarMineDataset(
            data_path=config['data_path'],
            years=config['years'],
            split_type="train",
            image_size=(512, 512),
            enhance_sonar=True,  # Use enhanced images for augmentation testing
            normalize=False,     # Don't normalize for visualization
            augmentations=None   # No augmentations yet
        )
        
        # Find samples with minimum required objects
        samples_with_objects = []
        for sample in dataset.samples:
            if sample['has_objects']:
                if sample['annotation_path'] and sample['annotation_path'].exists():
                    try:
                        with open(sample['annotation_path'], 'r') as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                            if len(lines) >= config['min_objects']:
                                samples_with_objects.append(sample)
                    except:
                        continue
        
        if len(samples_with_objects) == 0:
            raise ValueError("No images found with required minimum objects!")
        
        # Randomly select requested number of samples
        random.seed(42)  # Reproducible results
        num_samples = min(config['num_samples'], len(samples_with_objects))
        selected_samples = random.sample(samples_with_objects, num_samples)
        
        print(f"Found {len(samples_with_objects)} images with objects")
        print(f"Selected {len(selected_samples)} samples for augmentation comparison")
        
        return selected_samples
    
    def create_augmentation_pipeline(self, variant_config: Dict) -> A.Compose:
        """Create albumentations pipeline from config"""
        if not variant_config.get('enabled', False):
            return A.Compose([])  # No augmentations
        
        transforms = []
        
        # Flip transformations
        if variant_config.get('horizontal_flip', 0) > 0:
            transforms.append(A.HorizontalFlip(p=variant_config['horizontal_flip']))
        
        if variant_config.get('vertical_flip', 0) > 0:
            transforms.append(A.VerticalFlip(p=variant_config['vertical_flip']))
        
        # Rotation
        if variant_config.get('rotation_limit', 0) > 0:
            transforms.append(A.Rotate(
                limit=variant_config['rotation_limit'],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7
            ))
        
        # Brightness and contrast
        brightness_limit = variant_config.get('brightness_limit', 0)
        contrast_limit = variant_config.get('contrast_limit', 0)
        if brightness_limit > 0 or contrast_limit > 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.6
            ))
        
        # Blur
        blur_limit = variant_config.get('blur_limit', 0)
        if blur_limit > 0:
            transforms.append(A.OneOf([
                A.GaussianBlur(blur_limit=(1, blur_limit), p=1.0),
                A.MedianBlur(blur_limit=blur_limit, p=1.0),
            ], p=0.3))
        
        # Noise
        noise_limit = variant_config.get('noise_limit', 0)
        if noise_limit > 0:
            transforms.append(A.GaussNoise(
                var_limit=(0, noise_limit * 255 * 255),
                p=0.4
            ))
        
        # Advanced options if enabled
        advanced = self.config.get('advanced_options', {})
        
        if advanced.get('enable_shear', False):
            transforms.append(A.Affine(
                shear=(-advanced['shear_limit'], advanced['shear_limit']),
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                p=0.3
            ))
        
        if advanced.get('enable_elastic', False):
            transforms.append(A.ElasticTransform(
                alpha=advanced['elastic_alpha'],
                sigma=advanced['elastic_sigma'],
                alpha_affine=0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.2
            ))
        
        if advanced.get('enable_cutout', False):
            transforms.append(A.CoarseDropout(
                max_holes=advanced['cutout_holes'],
                max_height=advanced['cutout_size'],
                max_width=advanced['cutout_size'],
                fill_value=0,
                p=0.3
            ))
        
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels']
        ))
    
    def load_image_and_boxes(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, List[List], List[int]]:
        """Load image and parse YOLO annotations"""
        # Load image
        image = cv2.imread(str(sample['image_path']))
        if image is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Parse YOLO annotations
        boxes, labels = [], []
        if sample['annotation_path'] and sample['annotation_path'].exists():
            try:
                h, w = image.shape[:2]
                with open(sample['annotation_path'], 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Convert YOLO format to pascal_voc format (x1, y1, x2, y2)
                            x1 = (x_center - width/2) * w
                            y1 = (y_center - height/2) * h
                            x2 = (x_center + width/2) * w
                            y2 = (y_center + height/2) * h
                            
                            # Ensure valid bounding box
                            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                            if x2 > x1 and y2 > y1:  # Valid box
                                boxes.append([x1, y1, x2, y2])
                                labels.append(class_id)
            except Exception as e:
                print(f"Error reading annotations: {e}")
        
        return image, boxes, labels
    
    def evaluate_augmentation_quality(self, original_image: np.ndarray, 
                                    augmented_image: np.ndarray,
                                    original_boxes: List, 
                                    augmented_boxes: List) -> Dict[str, float]:
        """Evaluate quality of augmentation"""
        metrics = {}
        
        eval_config = self.config['evaluation']
        
        if eval_config.get('check_bbox_preservation', True):
            # Check if bounding boxes are preserved reasonably
            box_preservation = len(augmented_boxes) / max(len(original_boxes), 1)
            metrics['bbox_preservation_ratio'] = box_preservation
        
        if eval_config.get('calculate_intensity_stats', True):
            # Compare intensity statistics
            orig_mean = np.mean(original_image)
            aug_mean = np.mean(augmented_image)
            intensity_change = abs(aug_mean - orig_mean) / orig_mean if orig_mean > 0 else 0
            metrics['intensity_change_ratio'] = intensity_change
        
        if eval_config.get('measure_object_visibility', True):
            # Simple visibility metric based on contrast in bounding box regions
            def calculate_box_contrast(img, boxes):
                if not boxes:
                    return 0
                total_contrast = 0
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    roi = img[y1:y2, x1:x2] if len(img.shape) == 2 else img[y1:y2, x1:x2, 0]
                    if roi.size > 0:
                        total_contrast += np.std(roi)
                return total_contrast / len(boxes)
            
            # Convert to grayscale for contrast calculation
            orig_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY) if len(original_image.shape) == 3 else original_image
            aug_gray = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2GRAY) if len(augmented_image.shape) == 3 else augmented_image
            
            orig_contrast = calculate_box_contrast(orig_gray, original_boxes)
            aug_contrast = calculate_box_contrast(aug_gray, augmented_boxes)
            
            visibility_ratio = aug_contrast / orig_contrast if orig_contrast > 0 else 1.0
            metrics['object_visibility_ratio'] = visibility_ratio
        
        return metrics
    
    def visualize_single_comparison(self, sample: Dict[str, Any]) -> None:
        """Create comprehensive augmentation comparison for one image"""
        # Load image and annotations
        original_image, original_boxes, labels = self.load_image_and_boxes(sample)
        
        # Get augmentation variants
        variants = self.config['augmentation_variants']
        num_variants = len(variants)
        num_versions = self.config['visualization']['show_multiple_versions']
        
        # Create figure
        fig_size = self.config['visualization']['figure_size']
        fig, axes = plt.subplots(num_variants, num_versions + 1, figsize=fig_size)
        if num_variants == 1:
            axes = [axes]  # Make it consistent for indexing
        
        sample_name = Path(sample['image_path']).name
        fig.suptitle(f'Augmentation Comparison - {sample_name}', fontsize=16, weight='bold')
        
        metrics_summary = {}
        
        for variant_idx, (variant_name, variant_config) in enumerate(variants.items()):
            # Create augmentation pipeline
            aug_pipeline = self.create_augmentation_pipeline(variant_config)
            
            # Show original in first column
            axes[variant_idx][0].imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
            axes[variant_idx][0].set_title(f'{variant_name.title()}\\nOriginal', fontsize=12)
            axes[variant_idx][0].axis('off')
            
            # Draw original bounding boxes
            if self.config['visualization']['show_bounding_boxes']:
                for box, label in zip(original_boxes, labels):
                    x1, y1, x2, y2 = box
                    width, height = x2 - x1, y2 - y1
                    rect = plt.Rectangle((x1, y1), width, height,
                                       linewidth=2, edgecolor='red', facecolor='none')
                    axes[variant_idx][0].add_patch(rect)
                    class_name = "MILCO" if label == 0 else "NOMBO"
                    axes[variant_idx][0].text(x1, y1-5, class_name, color='red', fontsize=9, weight='bold')
            
            # Generate multiple augmented versions
            variant_metrics = []
            for version_idx in range(num_versions):
                try:
                    # Apply augmentation
                    if variant_config.get('enabled', False):
                        result = aug_pipeline(image=original_image, bboxes=original_boxes, class_labels=labels)
                        aug_image = result['image']
                        aug_boxes = result['bboxes']
                        aug_labels = result['class_labels']
                    else:
                        aug_image = original_image.copy()
                        aug_boxes = original_boxes.copy()
                        aug_labels = labels.copy()
                    
                    # Show augmented image
                    axes[variant_idx][version_idx + 1].imshow(aug_image, cmap='gray' if len(aug_image.shape) == 2 else None)
                    axes[variant_idx][version_idx + 1].set_title(f'Version {version_idx + 1}', fontsize=10)
                    axes[variant_idx][version_idx + 1].axis('off')
                    
                    # Draw augmented bounding boxes
                    if self.config['visualization']['show_bounding_boxes']:
                        for box, label in zip(aug_boxes, aug_labels):
                            x1, y1, x2, y2 = box
                            width, height = x2 - x1, y2 - y1
                            rect = plt.Rectangle((x1, y1), width, height,
                                               linewidth=2, edgecolor='lime', facecolor='none')
                            axes[variant_idx][version_idx + 1].add_patch(rect)
                            class_name = "MILCO" if label == 0 else "NOMBO"
                            axes[variant_idx][version_idx + 1].text(x1, y1-5, class_name, color='lime', fontsize=9, weight='bold')
                    
                    # Calculate metrics for this version
                    metrics = self.evaluate_augmentation_quality(original_image, aug_image, original_boxes, aug_boxes)
                    variant_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"Error applying augmentation {variant_name} version {version_idx + 1}: {e}")
                    # Show error placeholder
                    axes[variant_idx][version_idx + 1].text(0.5, 0.5, 'Augmentation\\nFailed', 
                                                           transform=axes[variant_idx][version_idx + 1].transAxes,
                                                           ha='center', va='center', fontsize=12)
                    axes[variant_idx][version_idx + 1].axis('off')
            
            # Store average metrics for this variant
            if variant_metrics:
                avg_metrics = {}
                for key in variant_metrics[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in variant_metrics])
                metrics_summary[variant_name] = avg_metrics
        
        plt.tight_layout()
        
        # Save results if requested
        if self.config['visualization']['save_results']:
            output_path = self.output_dir / f"augmentation_comparison_{Path(sample['image_path']).stem}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to: {output_path}")
        
        plt.show()
        
        # Print metrics summary
        print(f"\\n📊 Augmentation Quality Metrics for {sample_name}:")
        print("=" * 70)
        for variant_name, metrics in metrics_summary.items():
            print(f"\\n{variant_name.upper()}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name.replace('_', ' ').title()}: {value:.3f}")
    
    def simulate_training_batches(self) -> None:
        """Simulate training batches to show augmentation diversity"""
        if not self.config['training_simulation']['enabled']:
            return
        
        print("\\n🎯 Simulating Training Batches with Augmentations")
        print("=" * 60)
        
        # Get simulation config
        sim_config = self.config['training_simulation']
        samples = self.find_sample_images()
        
        # Use moderate augmentation for simulation
        moderate_config = self.config['augmentation_variants']['moderate']
        aug_pipeline = self.create_augmentation_pipeline(moderate_config)
        
        for batch_idx in range(sim_config['simulate_batches']):
            print(f"\\n📦 Batch {batch_idx + 1}/{sim_config['simulate_batches']}")
            
            # Create figure for this batch
            batch_size = sim_config['batch_size']
            cols = min(batch_size, 4)  # Max 4 columns
            rows = (batch_size + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
            if rows == 1 and cols == 1:
                axes = [[axes]]
            elif rows == 1:
                axes = [axes]
            elif cols == 1:
                axes = [[ax] for ax in axes]
            
            fig.suptitle(f'Training Batch {batch_idx + 1} - Augmentation Diversity', 
                        fontsize=14, weight='bold')
            
            for sample_idx in range(batch_size):
                row = sample_idx // cols
                col = sample_idx % cols
                
                # Select random sample
                sample = random.choice(samples)
                
                try:
                    # Load and augment
                    image, boxes, labels = self.load_image_and_boxes(sample)
                    result = aug_pipeline(image=image, bboxes=boxes, class_labels=labels)
                    aug_image = result['image']
                    aug_boxes = result['bboxes']
                    aug_labels = result['class_labels']
                    
                    # Show augmented image
                    axes[row][col].imshow(aug_image, cmap='gray' if len(aug_image.shape) == 2 else None)
                    axes[row][col].set_title(f'Sample {sample_idx + 1}', fontsize=10)
                    axes[row][col].axis('off')
                    
                    # Draw bounding boxes
                    for box, label in zip(aug_boxes, aug_labels):
                        x1, y1, x2, y2 = box
                        width, height = x2 - x1, y2 - y1
                        rect = plt.Rectangle((x1, y1), width, height,
                                           linewidth=1.5, edgecolor='yellow', facecolor='none')
                        axes[row][col].add_patch(rect)
                        class_name = "MILCO" if label == 0 else "NOMBO"
                        axes[row][col].text(x1, y1-3, class_name, color='yellow', fontsize=8, weight='bold')
                
                except Exception as e:
                    axes[row][col].text(0.5, 0.5, 'Failed', transform=axes[row][col].transAxes,
                                       ha='center', va='center')
                    axes[row][col].axis('off')
            
            # Hide unused subplots
            for sample_idx in range(batch_size, rows * cols):
                row = sample_idx // cols
                col = sample_idx % cols
                axes[row][col].axis('off')
            
            plt.tight_layout()
            
            if self.config['visualization']['save_results']:
                output_path = self.output_dir / f"training_batch_{batch_idx + 1}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            plt.show()
    
    def run_optimization(self) -> None:
        """Run the complete augmentation optimization workflow"""
        print("🚀 Data Augmentation Optimization Tool")
        print("=" * 50)
        
        # Find sample images
        samples = self.find_sample_images()
        
        # Process each sample
        for i, sample in enumerate(samples, 1):
            print(f"\\n🖼️  Processing sample {i}/{len(samples)}: {Path(sample['image_path']).name}")
            self.visualize_single_comparison(sample)
        
        # Simulate training batches if enabled
        if self.config['training_simulation']['enabled']:
            self.simulate_training_batches()
        
        # Print optimization guidelines
        self.print_optimization_guidelines()
    
    def print_optimization_guidelines(self) -> None:
        """Print guidelines for choosing optimal augmentation parameters"""
        print("\\n" + "=" * 80)
        print("🎯 AUGMENTATION PARAMETER OPTIMIZATION GUIDELINES")
        print("=" * 80)
        
        print("\\n📈 How to Choose Optimal Augmentation Parameters:")
        print("=" * 50)
        
        print("\\n1. GEOMETRIC TRANSFORMATIONS:")
        print("   • Horizontal Flip: 0.3-0.5 (sonar patterns can be directional)")
        print("   • Vertical Flip: 0.1-0.3 (usually less relevant for sonar)")
        print("   • Rotation: 10-20° (small angles, sonar usually aligned)")
        print("   • Look for: Objects remain recognizable, bounding boxes preserved")
        
        print("\\n2. INTENSITY TRANSFORMATIONS:")
        print("   • Brightness: 0.1-0.3 (sonar intensity varies)")
        print("   • Contrast: 0.1-0.2 (moderate changes)")
        print("   • Look for: Objects stay visible, not over/under-exposed")
        
        print("\\n3. NOISE AND BLUR:")
        print("   • Gaussian Noise: 0.02-0.05 (2-5%, sonar already noisy)")
        print("   • Blur: 3-5 kernel size (simulate sensor variations)")
        print("   • Look for: Realistic noise levels, objects still detectable")
        
        print("\\n4. EVALUATION METRICS:")
        print("   • Bbox Preservation: >0.9 excellent, >0.8 acceptable")
        print("   • Intensity Change: <0.3 good, <0.2 excellent")
        print("   • Object Visibility: >0.8 good, >0.9 excellent")
        
        print("\\n5. TRAINING CONSIDERATIONS:")
        print("   • Batch Diversity: Each batch should show varied augmentations")
        print("   • Consistency: Similar objects should augment similarly")
        print("   • Balance: Don't over-augment (model becomes too invariant)")
        
        print("\\n💡 RECOMMENDED STARTING POINTS:")
        print("   • Conservative: Use for high-quality datasets")
        print("   • Moderate: Best general-purpose choice")
        print("   • Sonar-Optimized: Tailored for underwater imagery")
        print("   • Aggressive: Use only if you have limited data")
        
        print("\\n🔧 PARAMETER TUNING PROCESS:")
        print("   1. Start with 'moderate' settings")
        print("   2. Check bbox_preservation_ratio > 0.8")
        print("   3. Ensure objects remain visually clear")
        print("   4. Test on validation set performance")
        print("   5. Adjust based on training stability")
        
        print(f"\\n📁 Results saved to: {self.output_dir}")
        print("   Compare augmentation effects to make informed decisions!")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Data Augmentation Optimization for Sonar Images'
    )
    parser.add_argument(
        '--config', '-c',
        default='augmentation_config.yaml',
        help='Path to augmentation configuration file'
    )
    parser.add_argument(
        '--simulate-training', '-s',
        action='store_true',
        help='Run training batch simulation only'
    )
    
    args = parser.parse_args()
    
    try:
        optimizer = AugmentationOptimizer(args.config)
        
        if args.simulate_training:
            optimizer.simulate_training_batches()
        else:
            optimizer.run_optimization()
        
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        print("Create augmentation_config.yaml or specify a different config file")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        print("Please check your configuration and data paths")


if __name__ == "__main__":
    main()