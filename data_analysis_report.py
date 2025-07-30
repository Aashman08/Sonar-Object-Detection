#!/usr/bin/env python3
"""
Sonar Mine Detection Dataset Analysis Script
Analyzes YOLO format annotations for object detection statistics
"""

import os
import glob
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class SonarDataAnalyzer:
    def __init__(self, dataset_path="."):
        self.dataset_path = Path(dataset_path)
        self.class_names = {0: "MILCO", 1: "NOMBO"}
        self.years = ["2010", "2015", "2017", "2018", "2021"]
        
        # Initialize statistics containers
        self.stats = {
            'overall': defaultdict(int),
            'by_year': defaultdict(lambda: defaultdict(int)),
            'objects_per_image': defaultdict(list),
            'class_distribution': defaultdict(int)
        }
    
    def parse_annotation_file(self, txt_file):
        """Parse a single YOLO format annotation file"""
        objects = []
        try:
            with open(txt_file, 'r') as f:
                lines = f.read().strip().split('\n')
                for line in lines:
                    if line.strip():  # Skip empty lines
                        parts = line.strip().split()
                        if len(parts) >= 5:  # class_id x y w h
                            class_id = int(parts[0])
                            objects.append(class_id)
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
        
        return objects
    
    def analyze_year_data(self, year):
        """Analyze data for a specific year"""
        year_path = self.dataset_path / year
        if not year_path.exists():
            print(f"Warning: Year {year} directory not found")
            return
        
        # Get all annotation files for this year
        txt_files = list(year_path.glob(f"*_{year}.txt"))
        jpg_files = list(year_path.glob(f"*_{year}.jpg"))
        
        year_stats = {
            'total_images': len(jpg_files),
            'total_annotations': len(txt_files),
            'images_with_objects': 0,
            'images_without_objects': 0,
            'total_objects': 0,
            'class_counts': defaultdict(int),
            'objects_per_image_dist': []
        }
        
        print(f"\nAnalyzing {year} data...")
        print(f"Found {len(txt_files)} annotation files and {len(jpg_files)} image files")
        
        for txt_file in txt_files:
            objects = self.parse_annotation_file(txt_file)
            num_objects = len(objects)
            
            if num_objects > 0:
                year_stats['images_with_objects'] += 1
                year_stats['total_objects'] += num_objects
                
                # Count objects by class
                for class_id in objects:
                    year_stats['class_counts'][class_id] += 1
                    self.stats['class_distribution'][class_id] += 1
            else:
                year_stats['images_without_objects'] += 1
            
            year_stats['objects_per_image_dist'].append(num_objects)
            self.stats['objects_per_image'][year].append(num_objects)
        
        # Store year statistics
        self.stats['by_year'][year] = year_stats
        
        # Update overall statistics
        self.stats['overall']['total_images'] += year_stats['total_images']
        self.stats['overall']['images_with_objects'] += year_stats['images_with_objects']
        self.stats['overall']['images_without_objects'] += year_stats['images_without_objects']
        self.stats['overall']['total_objects'] += year_stats['total_objects']
        
        return year_stats
    
    def generate_summary_table(self):
        """Generate a summary table by year"""
        rows = []
        
        for year in self.years:
            if year in self.stats['by_year']:
                year_data = self.stats['by_year'][year]
                object_rate = (year_data['images_with_objects'] / year_data['total_images'] * 100) if year_data['total_images'] > 0 else 0
                
                # Find the most common characteristic
                if object_rate < 15:
                    note = "Lowest object density"
                elif object_rate > 90:
                    note = "**Highest object density**"
                elif year_data['total_images'] > 400:
                    note = "**Largest dataset**"
                elif object_rate > 50:
                    note = "Small but rich"
                else:
                    note = "Medium density"
                
                rows.append({
                    'Year': year,
                    'Total Images': year_data['total_images'],
                    'With Objects': year_data['images_with_objects'],
                    'Object Rate': f"{object_rate:.1f}%",
                    'Notes': note
                })
        
        return pd.DataFrame(rows)
    
    def print_detailed_report(self):
        """Print comprehensive analysis report"""
        print("="*80)
        print("🔍 SONAR MINE DETECTION DATASET ANALYSIS REPORT")
        print("="*80)
        
        # Overall Statistics
        print("\n📊 DATASET OVERVIEW")
        print("-" * 40)
        total_images = self.stats['overall']['total_images']
        images_with_objects = self.stats['overall']['images_with_objects']
        images_without_objects = self.stats['overall']['images_without_objects']
        total_objects = self.stats['overall']['total_objects']
        
        print(f"• Total Images: {total_images:,}")
        print(f"• Time Span: 2010-2021 ({len(self.years)} years of data)")
        print(f"• Object Classes: 2 mine types")
        
        # Class distribution
        total_class_objects = sum(self.stats['class_distribution'].values())
        for class_id, count in self.stats['class_distribution'].items():
            percentage = (count / total_class_objects * 100) if total_class_objects > 0 else 0
            print(f"  - Class {class_id}: {self.class_names[class_id]} ({count} objects, {percentage:.1f}%)")
        
        print(f"• Data Distribution:")
        images_with_obj_pct = (images_with_objects / total_images * 100) if total_images > 0 else 0
        images_without_obj_pct = (images_without_objects / total_images * 100) if total_images > 0 else 0
        print(f"  - Images with objects: {images_with_objects} ({images_with_obj_pct:.1f}%)")
        print(f"  - Images without objects: {images_without_objects} ({images_without_obj_pct:.1f}%)")
        
        # Per-Year Breakdown
        print("\n📅 PER-YEAR BREAKDOWN")
        print("-" * 40)
        summary_df = self.generate_summary_table()
        print(summary_df.to_string(index=False))
        
        # Detailed Year Analysis
        print("\n🔬 DETAILED YEAR-BY-YEAR ANALYSIS")
        print("-" * 40)
        
        for year in self.years:
            if year in self.stats['by_year']:
                year_data = self.stats['by_year'][year]
                print(f"\n🗓️  {year} Dataset:")
                print(f"   • Total Images: {year_data['total_images']}")
                print(f"   • Images with Objects: {year_data['images_with_objects']}")
                print(f"   • Images without Objects: {year_data['images_without_objects']}")
                print(f"   • Total Objects: {year_data['total_objects']}")
                print(f"   • Average Objects per Image: {year_data['total_objects'] / year_data['total_images']:.2f}")
                
                if year_data['class_counts']:
                    print(f"   • Class Distribution:")
                    for class_id, count in year_data['class_counts'].items():
                        print(f"     - {self.class_names[class_id]}: {count} objects")
                
                # Objects per image statistics
                objects_per_img = year_data['objects_per_image_dist']
                if objects_per_img:
                    max_objects = max(objects_per_img)
                    avg_objects_with_obj = sum([x for x in objects_per_img if x > 0]) / len([x for x in objects_per_img if x > 0]) if any(x > 0 for x in objects_per_img) else 0
                    print(f"   • Max Objects in Single Image: {max_objects}")
                    print(f"   • Avg Objects per Image (with objects): {avg_objects_with_obj:.2f}")
        
        # Key Insights
        print("\n💡 KEY INSIGHTS")
        print("-" * 40)
        
        # Class imbalance
        class_counts = list(self.stats['class_distribution'].values())
        if len(class_counts) == 2:
            imbalance_ratio = max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')
            if imbalance_ratio < 2:
                imbalance_level = "Low"
            elif imbalance_ratio < 3:
                imbalance_level = "Moderate"
            else:
                imbalance_level = "High"
            print(f"• Class Imbalance: {imbalance_level} ({max(class_counts)/sum(class_counts)*100:.1f}% vs {min(class_counts)/sum(class_counts)*100:.1f}%)")
        
        # Complex scenes
        all_objects_per_image = []
        for year_objects in self.stats['objects_per_image'].values():
            all_objects_per_image.extend(year_objects)
        
        if all_objects_per_image:
            max_objects_scene = max(all_objects_per_image)
            print(f"• Complex Scenes: Up to {max_objects_scene} objects per image")
        
        # Negative samples
        negative_rate = (images_without_objects / total_images * 100) if total_images > 0 else 0
        print(f"• High Negative Rate: {negative_rate:.1f}% negative samples (good for hard negative mining)")
        
        # Recommendations
        print("\n🎯 RECOMMENDATIONS")
        print("-" * 40)
        
        # Find year with highest object density for training start
        best_year = None
        best_rate = 0
        for year in self.years:
            if year in self.stats['by_year']:
                year_data = self.stats['by_year'][year]
                rate = year_data['images_with_objects'] / year_data['total_images'] if year_data['total_images'] > 0 else 0
                if rate > best_rate:
                    best_rate = rate
                    best_year = year
        
        if best_year:
            print(f"• Start training with {best_year} data (highest object density: {best_rate*100:.1f}%)")
        
        # Find year with lowest object density for hard negatives
        worst_year = None
        worst_rate = 1.0
        for year in self.years:
            if year in self.stats['by_year']:
                year_data = self.stats['by_year'][year]
                rate = year_data['images_with_objects'] / year_data['total_images'] if year_data['total_images'] > 0 else 0
                if rate < worst_rate:
                    worst_rate = rate
                    worst_year = year
        
        if worst_year:
            print(f"• Use {worst_year} data for hard negative mining (lowest object density: {worst_rate*100:.1f}%)")
        
        print("• Implement data augmentation to balance class distribution")
        print("• Consider temporal splitting for train/val/test")
        print("• Standardize image resolutions during preprocessing")
    
    def create_visualizations(self):
        """Create data visualization plots"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sonar Mine Detection Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Images with/without objects by year
        years_data = []
        with_objects = []
        without_objects = []
        
        for year in self.years:
            if year in self.stats['by_year']:
                years_data.append(year)
                year_data = self.stats['by_year'][year]
                with_objects.append(year_data['images_with_objects'])
                without_objects.append(year_data['images_without_objects'])
        
        x = range(len(years_data))
        axes[0,0].bar(x, with_objects, label='With Objects', alpha=0.8, color='skyblue')
        axes[0,0].bar(x, without_objects, bottom=with_objects, label='Without Objects', alpha=0.8, color='lightcoral')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Number of Images')
        axes[0,0].set_title('Images Distribution by Year')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(years_data)
        axes[0,0].legend()
        
        # 2. Object density by year
        object_rates = []
        for year in years_data:
            year_data = self.stats['by_year'][year]
            rate = (year_data['images_with_objects'] / year_data['total_images'] * 100) if year_data['total_images'] > 0 else 0
            object_rates.append(rate)
        
        axes[0,1].bar(years_data, object_rates, color='lightgreen', alpha=0.8)
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Object Rate (%)')
        axes[0,1].set_title('Object Detection Rate by Year')
        axes[0,1].set_ylim(0, 100)
        
        # Add value labels on bars
        for i, rate in enumerate(object_rates):
            axes[0,1].text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Class distribution
        class_names = [self.class_names[i] for i in sorted(self.stats['class_distribution'].keys())]
        class_counts = [self.stats['class_distribution'][i] for i in sorted(self.stats['class_distribution'].keys())]
        
        colors = ['#ff9999', '#66b3ff']
        axes[1,0].pie(class_counts, labels=class_names, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1,0].set_title('Class Distribution (Overall)')
        
        # 4. Objects per image distribution
        all_objects_per_image = []
        for year_objects in self.stats['objects_per_image'].values():
            all_objects_per_image.extend(year_objects)
        
        # Create histogram of objects per image
        max_objects = max(all_objects_per_image) if all_objects_per_image else 0
        bins = range(0, max_objects + 2)
        axes[1,1].hist(all_objects_per_image, bins=bins, alpha=0.7, color='purple', edgecolor='black')
        axes[1,1].set_xlabel('Number of Objects per Image')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of Objects per Image')
        axes[1,1].set_xticks(range(0, max_objects + 1))
        
        plt.tight_layout()
        plt.savefig('sonar_dataset_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📈 Visualization saved as 'sonar_dataset_analysis.png'")
        
        return fig
    
    def export_statistics(self):
        """Export detailed statistics to CSV files"""
        # Export summary table
        summary_df = self.generate_summary_table()
        summary_df.to_csv('dataset_summary_by_year.csv', index=False)
        
        # Export detailed statistics
        detailed_stats = []
        for year in self.years:
            if year in self.stats['by_year']:
                year_data = self.stats['by_year'][year]
                for class_id, count in year_data['class_counts'].items():
                    detailed_stats.append({
                        'Year': year,
                        'Class_ID': class_id,
                        'Class_Name': self.class_names[class_id],
                        'Object_Count': count,
                        'Total_Images': year_data['total_images'],
                        'Images_With_Objects': year_data['images_with_objects']
                    })
        
        detailed_df = pd.DataFrame(detailed_stats)
        detailed_df.to_csv('detailed_class_statistics.csv', index=False)
        
        print(f"\n💾 Statistics exported:")
        print(f"   • dataset_summary_by_year.csv")
        print(f"   • detailed_class_statistics.csv")
    
    def run_analysis(self):
        """Run complete dataset analysis"""
        print("🚀 Starting Sonar Mine Detection Dataset Analysis...")
        print("=" * 60)
        
        # Analyze each year
        for year in self.years:
            self.analyze_year_data(year)
        
        # Generate comprehensive report
        self.print_detailed_report()
        
        # Create visualizations
        try:
            self.create_visualizations()
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
        
        # Export statistics
        try:
            self.export_statistics()
        except Exception as e:
            print(f"Warning: Could not export statistics: {e}")
        
        print("\n✅ Analysis Complete!")
        print("=" * 60)

def main():
    """Main function to run the analysis"""
    analyzer = SonarDataAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 