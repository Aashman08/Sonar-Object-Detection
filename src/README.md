# SonarMineDataset - PyTorch Dataset for Sonar Mine Detection

A comprehensive PyTorch dataset class for loading and preprocessing side scan sonar imagery for underwater mine detection tasks.

## 🚀 Features

### Core Functionality
- **YOLO Format Support**: Native parsing of YOLO format annotations (class_id x_center y_center width height)
- **Mixed Resolution Handling**: Smart resizing with aspect ratio preservation using letterboxing
- **Temporal Data Splitting**: Time-based train/val/test splits to prevent data leakage
- **Class Balancing**: Built-in strategies for handling imbalanced datasets

### Sonar-Specific Processing
- **Sonar Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
- **Noise Reduction**: Gaussian filtering to reduce sonar speckle noise
- **Intensity Normalization**: Proper handling of varying sonar intensities

### Data Augmentation
- **Geometric Augmentations**: Rotation, flipping, scaling (respecting sonar physics)
- **Photometric Augmentations**: Brightness, contrast, noise injection
- **Advanced Techniques**: Support for mixup, cutmix, and sonar-specific augmentations

### Production Ready
- **Comprehensive Validation**: Built-in data integrity checks
- **Memory Optimization**: Optional image caching and efficient loading
- **Error Handling**: Graceful handling of corrupted files and missing annotations
- **Export Capabilities**: Export annotations to CSV, COCO formats

## 📦 Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Or install specific packages
pip install torch torchvision opencv-python albumentations PyYAML pandas
```

## 🎯 Quick Start

### Basic Usage

```python
from sonar_dataset import SonarMineDataset

# Create a basic dataset
dataset = SonarMineDataset(
    data_path="./Data/",
    years=["2021"],
    split_type="train",
    image_size=(512, 512),
    enhance_sonar=True
)

print(f"Dataset created with {len(dataset)} samples")

# Load a sample
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Boxes: {sample['boxes']}")
print(f"Labels: {sample['labels']}")
```

### Temporal Splitting

```python
from sonar_dataset import TemporalSplitter

# Create temporal splits
splitter = TemporalSplitter(
    train_years=["2010", "2015", "2017"],
    val_years=["2018"],
    test_years=["2021"]
)

# Create datasets
datasets = splitter.create_splits(
    data_path="./Data/",
    image_size=(512, 512)
)

# Create DataLoaders
dataloaders = splitter.create_dataloaders(
    data_path="./Data/",
    batch_size=16,
    num_workers=4
)
```

### Configuration-Based Setup

```python
from sonar_dataset import SonarDatasetFactory

# Create from YAML configuration
datasets, dataloaders = SonarDatasetFactory.create_full_pipeline(
    config_path="config/dataset_config.yaml",
    batch_size=16,
    num_workers=4
)
```

## 📁 Data Structure Expected

```
Data/
├── 2010/
│   ├── 0001_2010.jpg
│   ├── 0001_2010.txt
│   ├── 0002_2010.jpg
│   └── 0002_2010.txt
├── 2015/
│   ├── 0001_2015.jpg
│   ├── 0001_2015.txt
│   └── ...
└── ...
```

### YOLO Annotation Format
Each `.txt` file contains one line per object:
```
class_id x_center y_center width height
```
Where all coordinates are normalized to [0, 1] range.

## ⚙️ Configuration

Create a YAML configuration file (see `config/dataset_config.yaml`):

```yaml
dataset:
  data_path: "./Data/"
  image_size: [512, 512]
  normalize: true
  enhance_sonar: true
  
  splits:
    temporal:
      train_years: ["2010", "2015", "2017"]
      val_years: ["2018"]
      test_years: ["2021"]
  
  class_mapping:
    0: "MILCO"
    1: "NOMBO"
  
  balance_strategy: null  # or "oversample_positive", "undersample_negative"
```

## 🔧 Advanced Features

### Custom Augmentations

```python
from sonar_dataset import get_training_augmentations

# Get pre-configured sonar augmentations
augmentations = get_training_augmentations((512, 512))

dataset = SonarMineDataset(
    data_path="./Data/",
    years=["2021"],
    augmentations=augmentations
)
```

### Dataset Validation

```python
# Validate dataset integrity
issues, stats = dataset.validate_dataset()

print(f"Validation statistics: {stats}")
if issues:
    print(f"Found {len(issues)} issues")
```

### Visualization

```python
# Visualize random samples
dataset.visualize_samples(num_samples=4)

# Export annotations
dataset.export_annotations("annotations.csv", format="csv")
```

### Class Balancing

```python
# Oversample positive examples
dataset = SonarMineDataset(
    data_path="./Data/",
    years=["2010", "2015"],
    balance_strategy="oversample_positive"
)

# Get class weights for loss function
class_weights = dataset.get_class_weights()
```

## 📊 Dataset Statistics

The dataset automatically computes comprehensive statistics:

```python
stats = dataset.get_statistics()
# Returns:
# {
#     'total_samples': 1170,
#     'positive_samples': 304,
#     'negative_samples': 866,
#     'positive_rate': 0.26,
#     'class_distribution': {0: 437, 1: 231},
#     'total_objects': 668,
#     'years': ['2010', '2015', '2017', '2018', '2021'],
#     'samples_per_year': {'2010': 345, '2015': 120, ...}
# }
```

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_dataset.py
```

For comprehensive examples:

```bash
python example_usage.py
```

## 🔍 DataLoader Integration

The dataset includes a custom collate function for handling variable-sized annotations:

```python
from torch.utils.data import DataLoader
from sonar_dataset import collate_fn

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

for images, targets in dataloader:
    # images: torch.Tensor of shape (batch_size, 3, height, width)
    # targets: List of dicts, one per image
    # Each target dict contains:
    #   - 'boxes': torch.Tensor of shape (num_objects, 4) [x1, y1, x2, y2]
    #   - 'labels': torch.Tensor of shape (num_objects,)
    #   - 'image_id': torch.Tensor
    #   - 'area': torch.Tensor of shape (num_objects,)
    #   - 'iscrowd': torch.Tensor of shape (num_objects,)
    pass
```

## 🎨 Visualization Output

The `visualize_samples()` method creates matplotlib plots showing:
- Original sonar images
- Bounding boxes with class labels
- Sample metadata (year, object count)

## 📤 Export Capabilities

Export annotations to different formats:

```python
# Export to CSV
dataset.export_annotations("annotations.csv", format="csv")

# CSV contains columns:
# image_path, year, has_objects, class_id, class_name, x1, y1, x2, y2, width, height
```

## 🚨 Error Handling

The dataset includes comprehensive error handling:
- **Corrupted Images**: Replaced with placeholder images
- **Missing Annotations**: Handled gracefully as negative samples
- **Invalid Boxes**: Automatically clipped to image bounds
- **Format Errors**: Logged with detailed error messages

## 🔧 Memory Optimization

```python
# Enable image caching for faster loading (requires more RAM)
dataset = SonarMineDataset(
    data_path="./Data/",
    cache_images=True  # Cache images in memory
)

# Use efficient DataLoader settings
dataloader = DataLoader(
    dataset,
    num_workers=4,        # Parallel loading
    pin_memory=True,      # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2     # Prefetch batches
)
```

## 🎯 Best Practices

### For Training
- Use temporal splitting to prevent data leakage
- Start with balanced subsets (e.g., 2015 data) for initial training
- Apply class balancing for imbalanced datasets
- Use progressive training: easy → hard examples

### For Validation
- Always validate dataset integrity before training
- Use separate years for validation to test temporal generalization
- Monitor class distribution across splits

### For Production
- Cache images if you have sufficient RAM
- Use appropriate number of workers based on your CPU
- Enable pin_memory for GPU training

## 🤝 Integration with Training Frameworks

### PyTorch

```python
# Direct integration
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

for images, targets in train_loader:
    # Your training loop
    pass
```

### Ultralytics YOLO

```python
# Convert format for Ultralytics
# The dataset can export YOLO format directly
dataset.export_annotations("yolo_format/", format="yolo")
```

## 📈 Performance Tips

1. **Use SSD storage** for faster I/O
2. **Optimize num_workers** based on your CPU cores
3. **Enable image caching** if you have enough RAM
4. **Use smaller image sizes** during development
5. **Profile your DataLoader** to identify bottlenecks

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Path Issues**: Check that `data_path` points to correct directory
3. **Memory Issues**: Reduce batch size or disable image caching
4. **Slow Loading**: Increase num_workers or enable caching

### Debug Mode

```python
# Create dataset with debug settings
dataset = SonarMineDataset(
    data_path="./Data/",
    years=["2021"],  # Small dataset
    cache_images=False,
    normalize=False  # For easier debugging
)

# Validate before use
issues, stats = dataset.validate_dataset()
```

## 📝 License

This dataset class is part of the Sonar Mine Detection project. See the main project README for license information.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Run `python test_dataset.py` to verify setup
3. Create an issue in the repository 