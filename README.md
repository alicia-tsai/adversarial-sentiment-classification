# Adversarial Sentiment Classification

Adversarial attack on sentiment classification task.

## Usage
#### Train Model
```
# datasets does not exists, build datasets from scratch
python run.py --build_data --alg CNN --outfile cnn-model.pt

# datasets already exists
python run.py --alg CNN --outfile cnn-model.pt
```

### Modules
#### Build and Save Datasets
```python
# Build and save datasets
from data_loader import DataLoader

data_loader = DataLoader()
data_loader.build_data()
```

#### Load Datasets
```python
from data_loader import DataLoader

data_loader = DataLoader()
data_loader.load_data()

# Dataset iterator
train_iter, valid_iter = data_loader.large_train_valid_iter()
```

#### Load Trained Model
```python
data_loader = DataLoader()
data_loader.load_data()

# This model is trained using large dataset
cnn_model = load_saved_model('CNN', 'saved_model/cnn-1.pt', data_loader)

# If model is trained using small datasets, set `small_subsets=True`
cnn_model = load_saved_model('CNN', 'saved_model/cnn-small.pt', data_loaderl, small_subsets=True)
```
