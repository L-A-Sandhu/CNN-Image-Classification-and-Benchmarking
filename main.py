

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.applications import (ConvNeXtXLarge, DenseNet201, EfficientNetB7, EfficientNetV2L, 
                                           InceptionResNetV2, MobileNetV3Large, NASNetLarge, RegNetX320, 
                                           ResNetRS420, VGG19, Xception)

from tensorflow.keras import backend as K
import tensorflow as tf
import json
import time
def clear_model(model):
    # Clear session, this will remove the model from memory
    K.clear_session()

    # Additionally, delete the model variable if it exists
    del model

    # Collect garbage (optional, can help in releasing memory faster)
    import gc
    gc.collect()

    # Manually clear TensorFlow's global graph (for TensorFlow 1.x compatibility)
    if tf.__version__.startswith('1.'):
        tf.reset_default_graph()

# Argument parser
parser = argparse.ArgumentParser(description='Train or test a CNN model.')
parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Choose to train or test the model.')
parser.add_argument('--model', type=str, choices=[
    'ConvNeXtXLarge', 'DenseNet201', 'EfficientNetB7', 'EfficientNetV2L', 
    'InceptionResNetV2', 'MobileNetV3Large', 'NASNetLarge', 'RegNetX320', 
    'ResNetRS420', 'VGG19', 'Xception'
], required=False, help='Specify the model architecture.')

parser.add_argument('--dataset', type=str, required=False, help='Specify the dataset name.')
parser.add_argument('--threshold', type=float, default=0.55, help='Specify the threshold for classifying predictions (default: 0.3)')
args = parser.parse_args()

# Function to generate class weights
def generate_class_weights(data_dir):
    labels = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            labels.extend([class_dir] * len([f for f in os.listdir(class_path) if f.endswith('.jpg')]))

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return {i: weight for i, weight in enumerate(class_weights)}

# Threshold for adjusting predictions
threshold = args.threshold

# Function to train the model
def train_model(dataset, model_choice):
    data_dir = f'./data/{dataset}/'
    train_data_dir = os.path.join(data_dir, 'train')
    class_weights = generate_class_weights(train_data_dir)
    data_dir = f'./data/{dataset}/'
    train_data_dir = os.path.join(data_dir, 'train')
    class_weights = generate_class_weights(train_data_dir)

    # Data generators for binary classification
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(128, 128),
        batch_size=8,
        class_mode='binary',
        subset='training'
    )

    valid_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(128, 128),
        batch_size=8,
        class_mode='binary',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(128, 128),
        batch_size=8,
        class_mode='binary',
        shuffle=False
    )
    if model_choice == 'ConvNeXtXLarge':
        # Assuming ConvNeXt models are available in your TensorFlow version
        # Replace with the correct import and model class if different
        from tensorflow.keras.applications import ConvNeXtXLarge
        preprocess_input = tf.keras.applications.convnext.preprocess_input
        base_model = ConvNeXtXLarge(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

    elif model_choice == 'DenseNet201':
        from tensorflow.keras.applications import DenseNet201
        preprocess_input = tf.keras.applications.densenet.preprocess_input
        base_model = DenseNet201(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

    elif model_choice == 'EfficientNetB7':
        from tensorflow.keras.applications import EfficientNetB7
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        base_model = EfficientNetB7(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

    elif model_choice == 'EfficientNetV2L':
        from tensorflow.keras.applications import EfficientNetV2L
        preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input
        base_model = EfficientNetV2L(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

    elif model_choice == 'InceptionResNetV2':
        from tensorflow.keras.applications import InceptionResNetV2
        preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
        base_model = InceptionResNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

    elif model_choice == 'MobileNetV3Large':
        from tensorflow.keras.applications import MobileNetV3Large
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
        base_model = MobileNetV3Large(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

    elif model_choice == 'NASNetLarge':
        from tensorflow.keras.applications import NASNetLarge
        preprocess_input = tf.keras.applications.nasnet.preprocess_input
        base_model = NASNetLarge(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

    elif model_choice == 'RegNetX320':
        from tensorflow.keras.applications import RegNetX320
        preprocess_input = tf.keras.applications.regnet.preprocess_input
        base_model = RegNetX320(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

    elif model_choice == 'ResNetRS420':
        from tensorflow.keras.applications import ResNetRS420
        preprocess_input = tf.keras.applications.resnet_rs.preprocess_input
        base_model = ResNetRS420(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

    elif model_choice == 'VGG19':
        from tensorflow.keras.applications import VGG19
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
        base_model = VGG19(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

    elif model_choice == 'Xception':
        from tensorflow.keras.applications import Xception
        preprocess_input = tf.keras.applications.xception.preprocess_input
        base_model = Xception(input_shape=(128, 128, 3), include_top=False, weights='imagenet')


    # Update data generators to use the selected model's preprocess_input
    train_datagen.preprocessing_function = preprocess_input
    test_datagen.preprocessing_function = preprocess_input

    # Create model directory
    model_dir = f'./checkpoint/{dataset}/{model_choice}/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define the rest of the model for binary classification
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)

    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint_path = os.path.join(model_dir, 'model.h5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=valid_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1,
        class_weight= class_weights 
    )

    # Save training history plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plot_path = os.path.join(model_dir, 'history_plot.png')
    plt.savefig(plot_path)

    # Evaluate the model on test data
    y_true = test_generator.classes
    y_pred_prob = model.predict(test_generator)
    
    # Adjust predictions using the threshold
    y_pred = (y_pred_prob >= args.threshold).astype(int)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Save evaluation results
    results_path = os.path.join(model_dir, 'Results.txt')
    with open(results_path, 'w') as results_file:
        results_file.write(f"Accuracy: {accuracy:.4f}\n")
        results_file.write(f"F1 Score: {f1:.4f}\n")
        results_file.write(f"Precision: {precision:.4f}\n")
        results_file.write(f"Recall: {recall:.4f}\n")
        results_file.write("Confusion Matrix:\n")
        results_file.write(np.array2string(conf_matrix, separator=', '))

    clear_model(model)

def test_model(dataset, model_choice):
    data_dir = f'./data/{dataset}/'
    test_data_dir = os.path.join(data_dir, 'test')

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    model_dir = f'./checkpoint/{dataset}/{model_choice}/'
    model_path = os.path.join(model_dir, 'model.h5')

    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Evaluate the model on test data
    y_true = test_generator.classes
    y_pred_prob = model.predict(test_generator)

    best_results = {
        "f1_score": 0
    }

    for threshold in np.arange(0.4, 0.7, 0.05):
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average='weighted')

        if f1 > best_results["f1_score"]:
            conf_matrix = confusion_matrix(y_true, y_pred)
            best_results = {
                "model": model_choice,
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_score": f1,
                "precision": precision_score(y_true, y_pred, average='weighted'),
                "recall": recall_score(y_true, y_pred, average='weighted'),
                "confusion_matrix": conf_matrix.tolist()
            }

    # Save evaluation results
    results_path = os.path.join(model_dir, 'Results.json')
    with open(results_path, 'w') as results_file:
        json.dump(best_results, results_file, indent=4)







if args.mode == 'train':
    selected_datasets = [args.dataset] if args.dataset else ['./data_1', './data_2']

    selected_models = [args.model] if args.model else [  'ConvNeXtXLarge', 'DenseNet201', 'EfficientNetB7', 'EfficientNetV2L', 
    'InceptionResNetV2', 'MobileNetV3Large', 'NASNetLarge', 'RegNetX320', 
    'ResNetRS420', 'VGG19', 'Xception'
    ]

    for dataset in selected_datasets:
        for model_choice in selected_models:
            model_dir = f"./checkpoint/{dataset}/{model_choice}/"  # Adjust the path according to your directory structure

            # Check if the model directory exists
            if not os.path.exists(model_dir):
                print("Model name", model_dir)
                train_model(dataset, model_choice)
            else:
                print(f"Model directory {model_dir} already exists. Skipping training.")

elif args.mode == 'test':
    selected_datasets = [args.dataset] if args.dataset else ['1_vs_all', '2_vs_all', '3_vs_all', '4_vs_all', '5_vs_all', '6_vs_all', '7_vs_all']
    selected_models = [args.model] if args.model else [ 'DenseNet201', 'EfficientNetB7', 
        'InceptionResNetV2', 'MobileNetV3Large', 'Xception'
    ]

    for dataset in selected_datasets:
        for model_choice in selected_models:
            test_model(dataset, model_choice)

