import tensorflow as tf
import argparse
from typing import Dict
from pathlib import Path, PureWindowsPath, PurePosixPath
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# References
# https://www.tensorflow.org/tutorials/images/classification
# the Tensorflow tutorial. Working to update this to work with
# Tensorflow 2.16.1...see completed tutorial elsewhere in this repo

# https://campus.datacamp.com/courses/introduction-to-deep-learning-with-keras/improving-your-model-performance
# https://github.com/jeffheaton/t81_558_deep_learning/blob/b59a6ca8334246a7b5358c81b9c7b1849bb45371/t81_558_class_06_2_cnn.ipynb

# TODO verify validation vs test
# TODO try larger image size
# TODO setup labeling of true and false classification results: see Jun's code
# TODO finish setting up arg parser

#################
# User Parameters
################

# Path to the ‘supervisor’ file.
#data_dir = r"C:\Users\josep\OneDrive - UCB-O365\Students\_shares\Lee HUB\junresearch\DeeplearningCNN_flow_detection\supervisor"

# Path to the output file
#output_file = r"C:\Users\josep\Documents\GitHub\MLtests\flow_detection\model_target680_batch16.keras"

GLOBAL_SETTINGS = {
    'batch_size': 24,
    'edge_size': 128,
    'epochs': 15,
    'learning_rate': 1e-3,
    'validation_split': 0.2,
    'trial_label': 'edge128batch24'
}

def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--computer',
        choices=['hpc', '2020laptop', 'seec_desktop'],
        type=str,
        help="Computer - dictates paths to supervisor data"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=False,
        help="Batch size"
    )
    parser.add_argument(
        '--edge_size',
        type=int,
        help="Edge size - for image resizing"
    )
    cfg = vars(parser.parse_args())

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    # insert validation checks here

    # add GPU device call here?

    return cfg

def set_supervisor_path(computer) -> Path:
    if computer == 'hpc':
        return Path("/projects/joka0958/supervisor/")
    elif computer == '2020laptop':
        return Path("C:/Users/josep/OneDrive - UCB-O365/Students/_shares/Lee HUB/junresearch/DeeplearningCNN_flow_detection/supervisor")

def set_output_path(computer) -> Path:
    if computer == 'hpc':
        return Path("/projects/joka0958/MLoutput/")
    elif computer == '2020laptop':
        return Path("C:/Users/josep/Documents/GitHub/MLtests/flow_detection/")


#################
# Model creation and testing
################

def main() -> None:
    config = get_args()

    supervisor_path = set_supervisor_path(config["computer"])
    output_path = set_output_path(config["computer"])

    # Create ImageDataGenerator to load and preprocess image data
    # note: this splits the data into two chunks
    # future task: split data into three chunks: train, validate, test

    # as per: https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
    # splitting this into two generators so that the training data are augmented
    # but the validation data are not

    train_ds = tf.keras.utils.image_dataset_from_directory(
        supervisor_path,
        validation_split=config['validation_split'],
        subset="training",
        seed=123,
        image_size=(config["edge_size"],
                    config["edge_size"]),
        batch_size=config["batch_size"]
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        supervisor_path,
        validation_split=config['validation_split'],
        subset="validation",
        seed=123,
        image_size=(config["edge_size"],
                    config["edge_size"]),
        batch_size=config["batch_size"]
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)


    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=config['validation_split']
    )

    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(config["edge_size"],
                                                config["edge_size"],
                                                3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # gradient accumulation steps?
    # different learning rate?
    # early stopping? https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_03_4_early_stop.ipynb)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    # model training
    history = model.fit(
        train_ds,
        epochs=config["epochs"],  # Set epochs count
        validation_data=val_ds,
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(config["epochs"])

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # Model evaluation and self-diagnosis
    #test_loss, test_accuracy = model.evaluate(val_data)
    #print(f'Test accuracy: {test_accuracy}')

    # Image classification prediction
    #predictions = model.predict(val_data)

    # Specify the path and file name to save the model C:/path/file_name.h5
    model.save(output_path / (config["trial_label"] + ".keras"))


if __name__ == '__main__':
    main()
