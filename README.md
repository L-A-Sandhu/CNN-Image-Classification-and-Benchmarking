# CNN Model Training and Testing Framework

This repository provides a comprehensive framework for training and testing Convolutional Neural Network (CNN) models for image classification tasks. The framework supports multiple state-of-the-art CNN architectures, data augmentation, class balancing, and evaluation of model performance with various metrics such as accuracy, F1-score, precision, and recall.

## Features
- **Multiple CNN Architectures**: Supports popular pre-trained models such as ConvNeXtXLarge, DenseNet201, EfficientNetB7, InceptionResNetV2, and many more.
- **Class Balancing**: Automatically generates class weights to handle imbalanced datasets.
- **Data Augmentation**: Includes advanced augmentation techniques like rotation, shear, zoom, and horizontal flips.
- **Callbacks**: Integrated support for early stopping, model checkpoints, and learning rate reduction.
- **Threshold Adjustment**: Fine-tune prediction thresholds to improve performance.
- **Training History Visualization**: Save training history plots for accuracy and loss.

## Supported Architectures
- ConvNeXtXLarge
- DenseNet201
- EfficientNetB7
- EfficientNetV2L
- InceptionResNetV2
- MobileNetV3Large
- NASNetLarge
- RegNetX320
- ResNetRS420
- VGG19
- Xception

## Requirements
- Python 3.x
- TensorFlow >= 2.x
- NumPy
- Matplotlib
- scikit-learn
- argparse
- json

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/L-A-Sandhu/<this-repo>.git

# Install the required dependencies:
pip install -r requirements.txt

# Ensure your dataset is structured as follows:
```
data/
├── dataset_name/
    ├── train/
        ├── class_1/
        ├── class_2/
    ├── test/
        ├── class_1/
        ├── class_2/
```

## Usage

### Training the Model

To train the model, run the following command:

~~~bash
python script_name.py --mode train --dataset <dataset_name> --model <model_name> --threshold <threshold_value>
~~~

### Command-Line Arguments:

- **`--mode`**: Operation mode. Use `train` for training.
- **`--dataset`**: Name of the dataset directory (inside the `data/` folder).
- **`--model`**: CNN architecture to use (e.g., `EfficientNetB7`, `DenseNet201`).
- **`--threshold`**: Threshold value for classification.


### Example:

~~~bash
python script_name.py --mode train --dataset cats_vs_dogs --model EfficientNetB7 --threshold 0.6
    
~~~

## Test the Model
 To test the model on a test dataset, use:

~~~bash
python script_name.py --mode test --dataset <dataset_name> --model <model_name>

~~~

### Example:

~~~bash
python script_name.py --mode test --dataset cats_vs_dogs --model EfficientNetB7
~~~

# Evaluation and Results
-The training process generates plots for accuracy and loss, saved in the ./checkpoint/<dataset>/<model>/history_plot.png.
-The evaluation metrics are saved in Results.txt and Results.json, including accuracy, F1-score, precision, recall, and the confusion matrix.
# Callbacks and Hyperparameter Tuning
-ModelCheckpoint: Automatically saves the best model based on validation accuracy.
-EarlyStopping: Stops training when the validation accuracy does not improve for 5 consecutive epochs.
ReduceLROnPlateau: Reduces the learning rate when validation loss stagnates.

# Customization
You can easily modify or add new models by including them in the train_model() and test_model() functions. Use pre-trained models from TensorFlow's keras.applications.

# Clear Model from Memory
To prevent memory issues when working with large models, the framework provides the clear_model() function that clears the model from memory after training or testing.
