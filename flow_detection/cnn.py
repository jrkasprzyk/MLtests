import tensorflow as tf
import argparse
from typing import Dict
from pathlib import Path, PureWindowsPath, PurePosixPath
#import matplotlib.pyplot as plt
import yaml
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from src.utils import set_output_path, set_supervisor_path

# References
# https://www.tensorflow.org/tutorials/images/classification
# the Tensorflow tutorial. Working to update this to work with
# Tensorflow 2.16.1...see completed tutorial elsewhere in this repo

# https://campus.datacamp.com/courses/introduction-to-deep-learning-with-keras/improving-your-model-performance
# https://github.com/jeffheaton/t81_558_deep_learning/blob/b59a6ca8334246a7b5358c81b9c7b1849bb45371/t81_558_class_06_2_cnn.ipynb

# TODO check Matplotlib in conda environment (error on seecdesktop); debug "import Sequential" error
# TODO continue learning about batch prediction and error labeling
# TODO implement cross validation

#################
# User Parameters
################

# Path to the ‘supervisor’ file.
#data_dir = r"C:\Users\josep\OneDrive - UCB-O365\Students\_shares\Lee HUB\junresearch\DeeplearningCNN_flow_detection\supervisor"

# Path to the output file
#output_file = r"C:\Users\josep\Documents\GitHub\MLtests\flow_detection\model_target680_batch16.keras"

#################
# Model creation and testing
################

def main() -> None:

    config = yaml.safe_load(open("config.yaml"))

    # show config
    print(config)

    supervisor_path = set_supervisor_path(config["computer"])
    output_path = set_output_path(config["computer"])

    # Create ImageDataGenerator to load and preprocess image data
    # note: this splits the data into two chunks
    # future task: split data into three chunks: train, validate, test

    # as per: https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
    # splitting this into two generators so that the training data are augmented
    # but the validation data are not

    # image_dataset_from_directory works differently than flow_from_directory
    # and this article suggests to skip it altogether and build your own
    # data pipeline:
    # https://medium.com/p/215e594f2435

    train_batched_ds, val_batched_ds = tf.keras.utils.image_dataset_from_directory(
        supervisor_path,
        validation_split=config['validation_split'],
        subset="both",
        seed=123,
        image_size=(config["edge_size"],
                    config["edge_size"]),
        batch_size=config["batch_size"]
    )

    # only the validation data, no batches
    val_single_ds = tf.keras.utils.image_dataset_from_directory(
        supervisor_path,
        validation_split=config['validation_split'],
        subset="validation",
        seed=123,
        image_size=(config["edge_size"],
                    config["edge_size"]),
        batch_size=None
    )

    class_names = val_batched_ds.class_names
    num_classes = len(class_names)

    model = Sequential([
        layers.Input(shape=(config["edge_size"], config["edge_size"], 3)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation=config["activation_function"]),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation=config["activation_function"]),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation=config["activation_function"]),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation=config["activation_function"]),
        layers.Dense(num_classes)
    ])

    # gradient accumulation steps?
    # different learning rate?
    # early stopping? https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_03_4_early_stop.ipynb)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    if config["verbose"]:
        model.summary()

    # model training
    history = model.fit(
        train_batched_ds,
        epochs=config["epochs"],  # Set epochs count
        validation_data=val_batched_ds,
    )

    # Save the model
    if config["save_model"]:
        model.save(output_path / (config["trial_label"] + ".keras"))

    # plot history
    if config["plot_history"]:
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

    # model.evaluate...including batching

    if config["verbose"]:
        print("WITH VALIDATION BATCHING")
        print("Evaluating model behavior...")

    val_loss, val_acc = model.evaluate(val_batched_ds)

    if config["verbose"]:
        print("loss=", val_loss, " acc=", val_acc)

    # see individual image prediction in cnn_from_file

if __name__ == '__main__':
    main()
