# Sonar Object Detection Dataset

A comprehensive PyTorch dataset pipeline for sonar mine detection using stratified data splitting and YAML-based configuration.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```python
cd DataPipeline
python main.py
```

This will:
- Load configuration from `../Config/sonar_dataset_config.yaml`
- Create stratified train/val/test splits
- Generate datasets and dataloaders
- Run validation tests
- Output ready-to-use dataloaders for training

## 📁 Data Structure Expected

```
Data/
├── 2010/
│   ├── 0001_2010.jpg
│   ├── 0001_2010.txt
│   ├── 0002_2010.jpg
│   └── 0002_2010.txt
├── 2015/
├── 2017/
├── 2018/
└── 2021/
```

**YOLO Annotation Format** (`.txt` files):
```
class_id x_center y_center width height
```
Where coordinates are normalized to [0, 1] range.

## ⚙️ Configuration

Configure everything via `../Config/sonar_dataset_config.yaml`:

```yaml
dataset:
  data_path: "./Data/"
  image_size: [512, 512]
  years: null  # null = all years, or specify ["2010", "2015"]
  
  # Data augmentation
  augmentations:
    enabled: true
    horizontal_flip: 0.5
    rotation_limit: 15
    brightness_contrast: 0.2
    noise_probability: 0.1
  
  # Stratified splitting
  splits:
    stratified:
      train_ratio: 0.7
      val_ratio: 0.15
      test_ratio: 0.15
      random_state: 42
  
  # Preprocessing
  normalize: true
  enhance_sonar: true
  balance_strategy: "oversample_positive"
  
  # CLAHE settings
  clahe_settings:
    clip_limit: 2.0
    tile_grid_size: [8, 8]

training:
  batch_size: 16
  num_workers: 4
```

## 🎯 Usage

### Configuration-Based (Recommended)
```python
from DataPipeline import SonarDatasetFactory

# Create complete pipeline from YAML config
datasets, dataloaders = SonarDatasetFactory.create_full_pipeline(
    config_path="../Config/sonar_dataset_config.yaml"
)

# Use in training
for images, targets in dataloaders['train']:
    # Your training code here
    pass
```

### Manual Usage
```python
from DataPipeline import StratifiedSplitter

splitter = StratifiedSplitter(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    balance_strategy='oversample_positive'
)

datasets = splitter.create_splits("./Data/", image_size=(512, 512))
dataloaders = splitter.create_dataloaders(datasets=datasets, batch_size=16)
```

## 🔧 Key Features

### Stratified Data Splitting
- Maintains class balance across splits
- Handles multi-year data correctly
- Reduces dataset imbalance (removes excess no-object images)

### Sonar-Specific Processing
- CLAHE enhancement for better contrast
- Configurable image preprocessing
- Physics-aware augmentations

### Extensible Augmentation Pipeline
- Add new augmentation parameters to YAML config
- Automatically handled by the pipeline
- Separate transforms for train/val/test

### Production Ready
- Memory-efficient DataLoaders
- Comprehensive error handling
- YAML-driven configuration
- Validation and statistics generation

## 📊 Class Information

- **Class 0**: MILCO (Mine-like objects)
- **Class 1**: NOMBO (Non-mine-like objects)
- **Class Balance**: Automatically handled via `balance_strategy`

## 🔍 Adding New Augmentations

Simply add to your YAML config:

```yaml
dataset:
  augmentations:
    enabled: true
    horizontal_flip: 0.5
    rotation_limit: 15
    brightness_contrast: 0.2
    new_augmentation: 0.3    # ← Add new params here
    another_param: true      # ← They'll work automatically
```

The pipeline will automatically pass these to the augmentation system.

## 📁 Project Structure

```
src/
├── Config/
│   └── sonar_dataset_config.yaml  # Configuration file
├── Data/                           # Your sonar imagery dataset
└── DataPipeline/                   # Complete data pipeline
    ├── main.py                     # Main entry point
    ├── DataProcessing/             # Core dataset pipeline
    │   ├── sonar_dataset_factory.py    # YAML-config factory
    │   ├── stratified_data_splitter.py # Stratified splitting
    │   ├── sonar_dataset.py            # PyTorch dataset class
    │   ├── sonar_augmentations.py      # Data augmentation pipeline
    │   └── README.md                   # Detailed data processing docs
    ├── Examples/                   # Usage examples
    └── Tests/                      # Test files
```

## 🚨 Troubleshooting

1. **Config not found**: Ensure `../Config/sonar_dataset_config.yaml` exists
2. **Data path issues**: Check `data_path` in config points to your Data directory  
3. **Memory issues**: Reduce `batch_size` or `num_workers` in config
4. **Import errors**: Run `pip install -r requirements.txt`
5. **Run from correct directory**: Execute `python main.py` from inside `DataPipeline/` folder

For detailed documentation on the data processing pipeline, see `DataProcessing/README.md`.