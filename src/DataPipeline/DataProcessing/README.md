# Data Processing Pipeline

Core data processing modules for sonar object detection dataset creation and management.

## 📦 Modules

### `sonar_dataset_factory.py`
YAML-configuration driven factory for creating complete dataset pipelines.

```python
from DataPipeline import SonarDatasetFactory

datasets, dataloaders = SonarDatasetFactory.create_full_pipeline(
    config_path="../Config/sonar_dataset_config.yaml"
)
```

### `stratified_data_splitter.py` 
Stratified data splitting that maintains class balance across train/val/test splits.

```python
from DataPipeline import StratifiedSplitter

splitter = StratifiedSplitter(
    train_ratio=0.7,
    val_ratio=0.15, 
    test_ratio=0.15,
    balance_strategy='oversample_positive'
)
```

### `sonar_dataset.py`
Core PyTorch Dataset class with sonar-specific preprocessing and YOLO annotation handling.

### `sonar_augmentations.py`
Configurable data augmentation pipeline using Albumentations.

```python
from DataPipeline.DataProcessing.sonar_augmentations import get_augmentations_from_config

transforms = get_augmentations_from_config(
    config={'horizontal_flip': 0.5, 'rotation_limit': 15},
    image_size=(512, 512),
    split_type='train'
)
```

### `dataset_utils.py`
Utility functions including custom collate functions for variable-sized annotations.

## 🔧 Key Features

### Stratified Splitting
- **Class Balance**: Maintains proportional class distribution across splits
- **Multi-year Support**: Handles data from different years (2010, 2015, 2017, 2018, 2021)
- **Dataset Balancing**: Automatically removes excess no-object images to balance dataset
- **Reproducible**: Uses fixed random seeds for consistent splits

### Configuration-Driven
- **YAML-Based**: All parameters controlled via configuration files
- **Extensible**: Add new parameters without code changes
- **Type Conversion**: Automatic handling of lists → tuples for compatibility
- **Smart Parameter Hierarchy**: Config values override constructor defaults
- **No Parameter Conflicts**: Clean parameter flow without popping or duplication

### Augmentation Pipeline
- **Split-Aware**: Different augmentations for train/val/test
- **Configurable**: Control all augmentation parameters via YAML
- **Extensible**: Easy to add new augmentation types
- **Sonar-Optimized**: Physics-aware transforms for underwater imagery

## 🎯 Data Flow

```
YAML Config → SonarDatasetFactory → StratifiedSplitter → SonarMineDataset
     ↓              ↓                    ↓                    ↓
  Parameters   Type Conversion    Sample Collection   Dataset Creation
     ↓              ↓                    ↓                    ↓
Augmentations → get_augmentations_from_config → Albumentations Pipeline
```

## ⚙️ Configuration Parameters

### Core Dataset Parameters
```yaml
dataset:
  data_path: "./Data/"                    # Path to data directory
  image_size: [512, 512]                  # Target image size [H, W]
  years: null                             # Years to use (null = all)
  normalize: true                         # Apply ImageNet normalization
  enhance_sonar: true                     # Apply CLAHE enhancement
  cache_images: false                     # Cache images in memory
  balance_strategy: "oversample_positive" # Class balancing strategy
```

### CLAHE Settings
```yaml
clahe_settings:
  clip_limit: 2.0                        # CLAHE clip limit
  tile_grid_size: [8, 8]                 # CLAHE tile grid size
```

### Stratified Splitting
```yaml
splits:
  stratified:
    train_ratio: 0.7                     # Training split ratio
    val_ratio: 0.15                      # Validation split ratio  
    test_ratio: 0.15                     # Test split ratio
    random_state: 42                     # Random seed
```

### Augmentation Configuration
```yaml
augmentations:
  enabled: true                          # Enable augmentations
  horizontal_flip: 0.5                   # Horizontal flip probability
  vertical_flip: 0.1                     # Vertical flip probability
  rotation_limit: 15                     # Max rotation degrees
  brightness_contrast: 0.2               # Brightness/contrast change limit
  contrast_limit: 0.2                    # Contrast change limit
  noise_probability: 0.1                 # Gaussian noise probability
```

## 🔍 Extending the Pipeline

### Adding New Augmentations

1. **Add to YAML config**:
```yaml
augmentations:
  new_augmentation: 0.3
  another_param: true
```

2. **Update `augmentations.py`**:
```python
def get_augmentations_from_config(config: dict, image_size: Tuple[int, int], split_type: str):
    # ...existing code...
    
    # Add your new augmentation
    if config.get('new_augmentation', 0) > 0:
        transforms.append(A.YourNewAugmentation(p=config['new_augmentation']))
```

### Adding New Dataset Parameters

Simply add to your YAML config - the factory will automatically pass all dataset parameters through:

```yaml
dataset:
  your_new_parameter: value              # Automatically passed through
  another_parameter: [1, 2, 3]          # Lists converted to tuples if needed
```

## 🎯 **Smart Parameter Hierarchy**

Our parameter system uses a clean hierarchy without conflicts:

1. **YAML Config Values** (highest priority)
2. **Constructor Defaults** (fallback)
3. **Hard-coded Defaults** (final fallback)

```python
# Example: balance_strategy parameter flow
balance_strategy = kwargs.get('balance_strategy', self.balance_strategy)
#                  ↑ from YAML config        ↑ from constructor
```

**Benefits:**
- ✅ **No parameter popping** - clean, immutable kwargs
- ✅ **Config overrides defaults** - YAML takes precedence
- ✅ **Explicit parameter handling** - easy to debug and extend
- ✅ **No conflicts** - each parameter handled once, cleanly

### Custom Splitting Strategies

Create new splitter classes following the `StratifiedSplitter` pattern:

```python
class CustomSplitter:
    def create_splits(self, data_path: str, **kwargs):
        # Your custom splitting logic
        return {'train': dataset1, 'val': dataset2, 'test': dataset3}
    
    def create_dataloaders(self, datasets: Dict, **kwargs):
        # Your custom DataLoader creation
        return {'train': loader1, 'val': loader2, 'test': loader3}
```

## 🧪 Class Distribution Analysis

The stratified splitter automatically analyzes your dataset:

- **MILCO (Class 0)**: Mine-like objects
- **NOMBO (Class 1)**: Non-mine-like objects  
- **No Objects**: Background images

**Balancing Strategy Options**:
- `"oversample_positive"`: Duplicate images with objects
- `"undersample_negative"`: Reduce background images  
- `null`: No balancing

## 📊 Sample Collection Process

1. **Multi-year Scanning**: Scans specified years (default: 2010, 2015, 2017, 2018, 2021)
2. **Annotation Parsing**: Reads YOLO format .txt files
3. **Stratification Keys**: Creates stratification based on:
   - `"no_object"`: Images with no annotations
   - `"mine_like"`: Images with MILCO objects (class 0)
   - `"non_mine_like"`: Images with NOMBO objects (class 1)
4. **Dataset Balancing**: Reduces no-object samples by 50% to balance dataset
5. **Split Creation**: Uses scikit-learn stratified splitting to maintain class balance

## 🔧 Technical Details

### Memory Optimization
- **Lazy Loading**: Images loaded on-demand
- **Optional Caching**: Set `cache_images: true` for faster access
- **Efficient DataLoaders**: Optimized batch loading with proper collation

### Error Handling
- **Missing Files**: Graceful handling of missing images/annotations
- **Corrupted Data**: Skip corrupted files with logging
- **Invalid Annotations**: Automatic bounding box validation and clipping

### Performance Tips
- **Parallel Loading**: Use `num_workers > 0` for faster data loading
- **Pin Memory**: Enable for GPU training
- **Batch Size**: Adjust based on available memory
- **Image Caching**: Enable if you have sufficient RAM

### Optimized Dataset Initialization
- **Pre-stratified Samples**: StratifiedSplitter pre-selects samples, SonarMineDataset skips file scanning
- **No Redundant Work**: Samples are already balanced and validated
- **Clear Logging**: "Dataset initialized with pre-stratified samples from StratifiedSplitter"
- **Faster Pipeline**: Eliminates duplicate file scanning and sample selection

## 🚨 Common Issues

1. **Memory Errors**: Reduce batch size or disable image caching
2. **Slow Loading**: Increase num_workers or enable caching  
3. **Import Errors**: Ensure all dependencies installed
4. **Config Errors**: Validate YAML syntax and required parameters
5. **Path Issues**: Check data_path points to correct directory

## 🔗 Integration

This pipeline integrates seamlessly with:
- **PyTorch**: Native DataLoader support
- **Lightning**: Easy integration with PyTorch Lightning
- **Weights & Biases**: Automatic logging of dataset statistics
- **MLflow**: Experiment tracking support