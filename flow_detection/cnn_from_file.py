import tensorflow as tf
from pathlib import Path
import numpy as np

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

def main() -> None:
    supervisor_path = set_supervisor_path("2020laptop")
    output_path = set_output_path("2020laptop")

    # load model from file

    # TODO what is difference between model.evaluate and model.predict?

    # set up validation data
    # note: will these parameters have to be the same as
    # the initial run? should they be saved so they can
    # be pulled in from a file?
    val_ds = tf.keras.utils.image_dataset_from_directory(
        supervisor_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(128,
                    128),
        batch_size=64
    )

    predictions = model.predict(val_ds)
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = val_ds.classes
    true_class_labels = val_ds.class_indices.keys()