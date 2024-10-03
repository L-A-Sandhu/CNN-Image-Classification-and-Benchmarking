# CNN Model Training and Testing Framework for Image Classification

This repository offers a complete framework for training, testing, and benchmarking Convolutional Neural Network (CNN) models for **image classification**. It supports multiple state-of-the-art **pre-trained CNN architectures** and provides tools for **data augmentation**, **class balancing**, and advanced **performance evaluation metrics** like accuracy, F1-score, precision, and recall. Ideal for researchers and developers looking to quickly test and benchmark CNN models.

## Key Features

- **Support for Multiple CNN Architectures**: Train and test with popular pre-trained models like ConvNeXtXLarge, DenseNet201, EfficientNetB7, InceptionResNetV2, and many more for image classification tasks.
- **Automatic Class Balancing**: Automatically generates class weights to manage imbalanced datasets and enhance CNN model accuracy.
- **Advanced Data Augmentation**: Apply augmentation techniques like rotation, shear, zoom, and horizontal flips to improve model generalization during image classification.
- **Integrated Callbacks for Efficient Training**: Supports early stopping, model checkpoints, and learning rate reduction to optimize deep learning model training.
- **Threshold Adjustment**: Fine-tune prediction thresholds to improve classification performance in deep learning models.
- **Training History Visualization**: Automatically save training history plots for accuracy and loss metrics, enabling easy performance monitoring.

## Supported Architectures

This framework supports multiple **state-of-the-art CNN architectures**, allowing users to train and test models on various pre-trained architectures known for **image classification** performance. The following architectures are supported:

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

To use this **deep learning framework**, ensure that you have the following **dependencies** installed:

- **Python 3.x**: Required for executing the framework's scripts and libraries.
- **TensorFlow >= 2.x**: The main deep learning library used for training CNN models.
- **NumPy**: For efficient numerical computation and data handling.
- **Matplotlib**: Used to visualize the **training history** (accuracy and loss) during model training.
- **scikit-learn**: Provides tools for model evaluation, including metrics like accuracy, F1-score, precision, and recall.
- **argparse**: Enables easy command-line argument parsing for flexibility in running the scripts.
- **json**: Required for handling configuration files and model parameters.

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

## Evaluation and Results

- **Accuracy and Loss Plots**: The training process automatically generates plots for accuracy and loss, which are saved in the directory: `./checkpoint/<dataset>/<model>/history_plot.png`. These plots provide a visual representation of the model's training performance.

- **Comprehensive Evaluation Metrics**: After testing, detailed evaluation metrics are saved in `Results.txt` and `Results.json`. These include:
    - **Accuracy**: Measures the overall percentage of correct predictions.
    - **F1-score**: Balances precision and recall, especially useful for imbalanced datasets.
    - **Precision**: The proportion of positive identifications that are actually correct.
    - **Recall**: The proportion of actual positives that are identified correctly.
    - **Confusion Matrix**: Provides a detailed breakdown of model predictions across all classes.

These metrics help you thoroughly **evaluate your CNN model’s performance** on the test dataset.
## Callbacks and Hyperparameter Tuning

This framework includes **integrated callbacks** for better control of the training process and **hyperparameter tuning** to optimize CNN model performance:

- **ModelCheckpoint**: Automatically saves the best model during training based on validation accuracy, ensuring that the most effective version of the model is retained for testing.
  
- **EarlyStopping**: Stops training when the validation accuracy does not improve for 5 consecutive epochs, preventing overfitting and saving computational resources by halting unnecessary training.

- **ReduceLROnPlateau**: Automatically reduces the learning rate when validation loss stagnates, helping the model to converge more effectively during the later stages of training.

These callbacks provide essential tools for **CNN model optimization**, helping to achieve better results with fewer resources.


## Customization

This framework is highly flexible and allows you to easily customize or add new CNN models for your specific tasks. Simply modify the `train_model()` and `test_model()` functions to integrate additional architectures from **TensorFlow’s Keras applications** or custom-built models.

- **How to Add New Models**: To include a new pre-trained model, import it from `keras.applications` and specify the desired input shapes and parameters. You can also fine-tune existing models by adding your own layers on top of the pre-trained base models.
  
- **Customization of Training and Testing Scripts**: You can modify the training or testing scripts to experiment with different **hyperparameters**, **loss functions**, and **optimization techniques**.

This flexibility allows you to tailor the framework for **custom CNN models**, making it ideal for research or **specialized deep learning tasks**.


## Clear Model from Memory

When working with large-scale **CNN models** and **high-dimensional datasets**, memory management becomes crucial. To avoid memory overflow and ensure smooth execution, this framework provides a `clear_model()` function, which safely clears the model from memory after training or testing.

- **When to Use It**: Call this function after each training or testing session, especially when using large models like ConvNeXtXLarge or ResNetRS420, to free up GPU and RAM resources.
  
- **Why It’s Important**: Proper memory management helps prevent crashes, reduces hardware strain, and allows you to experiment with different models without restarting your session.

This feature is essential for handling **large deep learning models** and maintaining efficient workflows.

## Frequently Asked Questions (FAQ)

**Q1: What CNN architectures are supported by this framework?**
A: This framework supports a wide range of **pre-trained CNN architectures** including ConvNeXtXLarge, EfficientNetB7, DenseNet201, InceptionResNetV2, and more. You can find the full list in the **Supported Architectures** section above.

**Q2: Can I use this framework for image classification on custom datasets?**
A: Yes! The framework is designed for flexibility and can be used with any custom dataset for **image classification**. Simply structure your dataset as outlined in the **Setup and Installation** section, and the framework will handle the rest.

**Q3: How does this framework handle imbalanced datasets?**
A: The framework automatically generates **class weights** to address class imbalance during training, ensuring that minority classes are properly accounted for during **CNN model training**.

**Q4: What evaluation metrics are included in the framework?**
A: After testing the model, the framework provides detailed evaluation metrics including **accuracy**, **F1-score**, **precision**, **recall**, and a **confusion matrix**. These metrics help assess the performance of your model on the test dataset.

**Q5: Can I add new CNN models to the framework?**
A: Absolutely! The framework is fully customizable. You can modify the `train_model()` and `test_model()` functions to add new **CNN architectures** or fine-tune pre-existing models using **TensorFlow’s Keras applications**.

**Q6: How can I visualize the model’s training history?**
A: The framework automatically saves **training history plots** for accuracy and loss in the `./checkpoint/<dataset>/<model>/history_plot.png` directory, making it easy to track and visualize model performance over time.

## Conclusion

This **CNN model training and testing framework** is designed to simplify the process of building, training, and evaluating CNN models for **image classification** tasks. With support for state-of-the-art architectures, advanced features like **class balancing** and **data augmentation**, and detailed performance metrics, this framework is ideal for both research and production-level projects.

Feel free to **fork the repository**, experiment with different CNN models, and contribute to the project. We welcome suggestions, improvements, and new features!

Don’t forget to ⭐ the repository if you find it useful.

