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

    my_filename = output_path / "edge64test.keras"

    # load model from file
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    model = tf.keras.models.load_model(my_filename)

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
        image_size=(64,
                    64),
        batch_size=64
    )

    print("beginning evaluation")
    my_loss, my_acc = model.evaluate(val_ds)
    print("loss=", my_loss, " acc=", my_acc)

    print("beginning prediction")
    predictions = model.predict(val_ds, verbose=2)
    predicted_classes = np.argmax(predictions, axis=1)
    print("predictions complete")

    # based on:
    # another tutorial that shows off different functions
    # https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/hub/tutorials/tf2_image_retraining.ipynb#scrollTo=umB5tswsfTEQ
    # and:
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    # which discusses things like rebatch and other actions on a Data object...

    print("pull one image")
    x, y = next(iter(val_ds.rebatch(1)))
    image = x[0, :, :, :]
    true_index = np.argmax(y[0])
    prediction_scores = model.predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print("True class: ", true_index)
    print("Predicted class: ", predicted_index)

    pass

    # in old version, this was:
    #true_classes = val_ds.classes
    #true_class_labels = val_ds.class_indices.keys()

    #true_classes = val_ds.class_func
    #print(true_classes)

    # this seems to discuss how to manipulate data in tensorflow 2.16.1
    # https://www.tensorflow.org/tutorials/load_data/images
    # problem is: I don't understand how to get the basic data extracted from val_ds
    # in other words: what is the true class for each image? What is the predicted class for each image?

    # see here for objective functions:
    # https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_04_2_multi_class.ipynb

if __name__ == "__main__":
    main()
