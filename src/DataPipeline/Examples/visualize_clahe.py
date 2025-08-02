#!/usr/bin/env python3
"""
CLAHE Enhancement Visualization Tool
====================================

Intelligent tool for comparing different CLAHE parameters on sonar images.
Helps you find optimal enhancement settings for your mine detection dataset.

Usage:
    python visualize_clahe.py                    # Use default config
    python visualize_clahe.py --config custom.yaml  # Use custom config
    
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
from collections import Counter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataProcessing import SonarMineDataset


class CLAHEOptimizer:
    """Smart CLAHE parameter optimization for sonar images"""
    
    def __init__(self, config_path: str = "YAML_Config/clahe_config.yaml"):
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
            enhance_sonar=False,  # We'll do enhancement manually
            normalize=False
        )
        
        # Find samples with minimum required objects
        samples_with_objects = []
        for sample in dataset.samples:
            if sample['has_objects']:
                # Quick check - count objects in annotation
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
        print(f"Selected {len(selected_samples)} samples for CLAHE comparison")
        
        return selected_samples
    
    def apply_clahe_variant(self, image: np.ndarray, variant_config: Dict) -> np.ndarray:
        """Apply CLAHE with specific parameters"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=variant_config['clip_limit'],
            tileGridSize=tuple(variant_config['tile_grid_size'])
        )
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur for noise reduction if specified
        blur_kernel = self.config['processing']['gaussian_blur_kernel']
        if blur_kernel > 0:
            enhanced = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)
        
        # Normalize if requested
        if self.config['processing']['normalize_output']:
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return enhanced
    
    def calculate_image_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """Calculate enhancement quality metrics"""
        metrics = {}
        
        if self.config['evaluation']['calculate_contrast']:
            # Contrast using standard deviation
            orig_contrast = np.std(original)
            enh_contrast = np.std(enhanced)
            metrics['contrast_improvement'] = enh_contrast / orig_contrast if orig_contrast > 0 else 1.0
        
        if self.config['evaluation']['calculate_sharpness']:
            # Sharpness using Laplacian variance
            orig_sharpness = cv2.Laplacian(original, cv2.CV_64F).var()
            enh_sharpness = cv2.Laplacian(enhanced, cv2.CV_64F).var()
            metrics['sharpness_improvement'] = enh_sharpness / orig_sharpness if orig_sharpness > 0 else 1.0
        
        if self.config['evaluation']['calculate_noise_level']:
            # Noise estimation using high-frequency content
            # Higher values indicate more noise
            kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
            orig_noise = np.mean(np.abs(cv2.filter2D(original, -1, kernel)))
            enh_noise = np.mean(np.abs(cv2.filter2D(enhanced, -1, kernel)))
            metrics['noise_ratio'] = enh_noise / orig_noise if orig_noise > 0 else 1.0
        
        return metrics
    
    def visualize_comparison(self, sample: Dict[str, Any]) -> None:
        """Create comprehensive CLAHE comparison visualization"""
        # Load original image
        original_image = cv2.imread(str(sample['image_path']))
        if original_image is None:
            print(f"Failed to load image: {sample['image_path']}")
            return
        
        # Convert to grayscale for processing
        if len(original_image.shape) == 3:
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original_image.copy()
        
        # Load annotations for visualization
        boxes, labels = [], []
        if sample['annotation_path'] and sample['annotation_path'].exists():
            try:
                h, w = original_gray.shape[:2]
                with open(sample['annotation_path'], 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Convert YOLO format to pixel coordinates
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id)
            except Exception as e:
                print(f"Error reading annotations: {e}")
        
        # Apply all CLAHE variants
        variants = self.config['clahe_variants']
        enhanced_images = {}
        metrics_data = {}
        
        for variant_name, variant_config in variants.items():
            enhanced = self.apply_clahe_variant(original_gray, variant_config)
            enhanced_images[variant_name] = enhanced
            metrics_data[variant_name] = self.calculate_image_metrics(original_gray, enhanced)
        
        # Create visualization
        fig_size = self.config['visualization']['figure_size']
        num_variants = len(variants)
        rows = 2 if self.config['visualization']['show_histograms'] else 1
        cols = num_variants + 1  # +1 for original
        
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        if rows == 1:
            axes = [axes]  # Make it consistent for indexing
        
        # Original image (first column)
        axes[0][0].imshow(original_gray, cmap='gray')
        axes[0][0].set_title('Original Image', fontsize=14, weight='bold')
        axes[0][0].axis('off')
        
        # Draw bounding boxes on original
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            rect = plt.Rectangle((x1, y1), width, height,
                               linewidth=2, edgecolor='red', facecolor='none')
            axes[0][0].add_patch(rect)
            class_name = "MILCO" if label == 0 else "NOMBO"
            axes[0][0].text(x1, y1-5, class_name, color='red', fontsize=10, weight='bold')
        
        # Show histogram for original if requested
        if self.config['visualization']['show_histograms']:
            axes[1][0].hist(original_gray.flatten(), bins=50, alpha=0.7, color='blue')
            axes[1][0].set_title('Original Histogram')
            axes[1][0].set_xlabel('Intensity')
            axes[1][0].set_ylabel('Frequency')
        
        # Enhanced variants (remaining columns)
        for idx, (variant_name, enhanced_image) in enumerate(enhanced_images.items(), 1):
            # Show enhanced image
            axes[0][idx].imshow(enhanced_image, cmap='gray')
            
            # Create title with parameters and metrics
            variant_config = variants[variant_name]
            title = f"{variant_name.title()}\n"
            title += f"Clip: {variant_config['clip_limit']}, Grid: {variant_config['tile_grid_size']}"
            
            if self.config['evaluation']['calculate_contrast']:
                contrast_imp = metrics_data[variant_name]['contrast_improvement']
                title += f"\nContrast: {contrast_imp:.2f}x"
            
            axes[0][idx].set_title(title, fontsize=12)
            axes[0][idx].axis('off')
            
            # Draw bounding boxes on enhanced image
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                width, height = x2 - x1, y2 - y1
                rect = plt.Rectangle((x1, y1), width, height,
                                   linewidth=2, edgecolor='lime', facecolor='none')
                axes[0][idx].add_patch(rect)
                class_name = "MILCO" if label == 0 else "NOMBO"
                axes[0][idx].text(x1, y1-5, class_name, color='lime', fontsize=10, weight='bold')
            
            # Show histogram for enhanced image if requested
            if self.config['visualization']['show_histograms']:
                axes[1][idx].hist(enhanced_image.flatten(), bins=50, alpha=0.7, color='green')
                axes[1][idx].set_title(f'{variant_name.title()} Histogram')
                axes[1][idx].set_xlabel('Intensity')
                axes[1][idx].set_ylabel('Frequency')
        
        # Overall title
        sample_name = Path(sample['image_path']).name
        fig.suptitle(f'CLAHE Enhancement Comparison - {sample_name}', 
                    fontsize=16, weight='bold')
        
        plt.tight_layout()
        
        # Save results if requested
        if self.config['visualization']['save_results']:
            output_path = self.output_dir / f"clahe_comparison_{Path(sample['image_path']).stem}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to: {output_path}")
        
        plt.show()
        
        # Print detailed metrics if requested
        if self.config['visualization']['show_statistics']:
            print(f"\n📊 Enhancement Statistics for {sample_name}:")
            print("=" * 60)
            for variant_name, metrics in metrics_data.items():
                print(f"\n{variant_name.upper()}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name.replace('_', ' ').title()}: {value:.3f}")
    
    def run_optimization(self) -> None:
        """Run the complete CLAHE optimization workflow"""
        print("🔬 CLAHE Enhancement Optimization Tool")
        print("=" * 50)
        
        # Find sample images
        samples = self.find_sample_images()
        
        # Process each sample
        for i, sample in enumerate(samples, 1):
            print(f"\n🖼️  Processing sample {i}/{len(samples)}: {Path(sample['image_path']).name}")
            self.visualize_comparison(sample)
        
        # Print optimization guidelines
        self.print_optimization_guidelines()
    
    def print_optimization_guidelines(self) -> None:
        """Print guidelines for choosing optimal CLAHE parameters"""
        print("\n" + "=" * 70)
        print("🎯 CLAHE PARAMETER OPTIMIZATION GUIDELINES")
        print("=" * 70)
        
        print("\n📈 How to Choose Optimal Parameters:")
        print("=" * 40)
        
        print("\n1. CLIP LIMIT (Controls contrast enhancement):")
        print("   • 1.0-2.0: Conservative, minimal artifacts")
        print("   • 2.0-3.0: Moderate, good balance")  
        print("   • 3.0-5.0: Aggressive, may introduce noise")
        print("   • Look for: Objects become more visible without excessive noise")
        
        print("\n2. TILE GRID SIZE (Controls local adaptation):")
        print("   • [4,4]: Coarse, global-like enhancement")
        print("   • [8,8]: Balanced, most commonly used")
        print("   • [16,16]: Fine, high local adaptation")
        print("   • Look for: Good object contrast without tile artifacts")
        
        print("\n3. EVALUATION METRICS:")
        print("   • Contrast Improvement: Higher = better object visibility")
        print("   • Sharpness Improvement: >1.0 good, >1.5 excellent")
        print("   • Noise Ratio: <1.5 acceptable, <1.2 excellent")
        
        print("\n4. VISUAL INSPECTION:")
        print("   • Objects should be clearly distinguishable")
        print("   • Background shouldn't be overly noisy")
        print("   • Histograms should show better distribution")
        print("   • Bounding boxes should align well")
        
        print("\n💡 RECOMMENDED STARTING POINTS:")
        print("   • Sonar imagery: clip_limit=2.0, tile_grid_size=[8,8]")
        print("   • High noise: clip_limit=1.5, tile_grid_size=[4,4]")
        print("   • Low contrast: clip_limit=3.0, tile_grid_size=[8,8]")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print("   Compare images side-by-side to make final decision!")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='CLAHE Enhancement Optimization for Sonar Images'
    )
    parser.add_argument(
        '--config', '-c',
        default='clahe_config.yaml',
        help='Path to CLAHE configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        optimizer = CLAHEOptimizer(args.config)
        optimizer.run_optimization()
        
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        print("Create clahe_config.yaml or specify a different config file")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        print("Please check your configuration and data paths")


if __name__ == "__main__":
    main()