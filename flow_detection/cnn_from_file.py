import tensorflow as tf
from pathlib import Path
import numpy as np

from src.utils import set_output_path, set_supervisor_path

def main() -> None:
    supervisor_path = set_supervisor_path("seecdesktop")
    output_path = set_output_path("seecdesktop")

    my_filename = output_path / "edge256batch128.keras"

    # load model from file
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    model = tf.keras.models.load_model(my_filename)

    # set up validation data
    # note: will these parameters have to be the same as
    # the initial run? should they be saved so they can
    # be pulled in from a file?
    val_single_ds = tf.keras.utils.image_dataset_from_directory(
        supervisor_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(256,
                    256),
        batch_size=128
    )

    # based on:
    # another tutorial that shows off different functions
    # https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/hub/tutorials/tf2_image_retraining.ipynb#scrollTo=umB5tswsfTEQ
    # and:
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    # which discusses things like rebatch and other actions on a Data object...

    print("pull one image")
    x, y = next(iter(val_single_ds))
    image = x[0, :, :, :]
    true_index = np.argmax(y[0])
    prediction_scores = model.predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print("True class: ", true_index)
    print("Predicted class: ", predicted_index)

    print("and again")
    x, y = next(iter(val_single_ds))
    image = x[0, :, :, :]
    true_index = np.argmax(y[0])
    prediction_scores = model.predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print("True class: ", true_index)
    print("Predicted class: ", predicted_index)

    pass

    # this seems to discuss how to manipulate data in tensorflow 2.16.1
    # https://www.tensorflow.org/tutorials/load_data/images
    # problem is: I don't understand how to get the basic data extracted from val_ds
    # in other words: what is the true class for each image? What is the predicted class for each image?

    # see here for objective functions:
    # https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_04_2_multi_class.ipynb

    # challenge: seems like there is still no straightforward way to pull the true class
    # for each datapoint when the data are loaded in like this
    # also: do the predictions always have to be done in batches?

if __name__ == "__main__":
    main()
