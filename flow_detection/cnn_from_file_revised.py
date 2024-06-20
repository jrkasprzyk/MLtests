import tensorflow as tf
from src.utils import set_output_path, set_supervisor_path
import numpy as np

print(tf.__version__)

supervisor_path = set_supervisor_path("2020laptop")
output_path = set_output_path("2020laptop")

my_filename = output_path / "edge256batch128.keras"

# load model from file
# https://www.tensorflow.org/tutorials/keras/save_and_load
model = tf.keras.models.load_model(my_filename)

# based on the latest keras training: https://www.tensorflow.org/guide/data
# we can manipulate Dataset objects and perform transforms on them if we need them
# so for example we can create a basic dataset and then augment it, batch it, etc.
# later on
train_unbatched_ds, val_unbatched_ds = tf.keras.utils.image_dataset_from_directory(
    supervisor_path,
    validation_split=0.05, # typically 0.2, but made smaller to make this example quick
    subset="both",
    seed=123,
    image_size=(256,256),
    batch_size=None
)

train_batched_ds = train_unbatched_ds.batch(32, drop_remainder=True)
val_batched_ds = val_unbatched_ds.batch(32, drop_remainder=True)

#print("Evaluating model behavior...")
#val_loss, val_acc = model.evaluate(val_batched_ds)
#print("loss=", val_loss, " acc=", val_acc)

print("Preparing confusion matrix")
# to prepare the confusion matrix, we are logging the model's performance
# on each datapoint individually
predictions = np.array([])
labels = np.array([])

for x, y in val_unbatched_ds:
    # to handle the fact the model was trained with batches and thus expects the first dimension to be 'None', see here:
    # https://stackoverflow.com/questions/60486437/add-none-dimension-in-tensorflow-2-0

    # also note that predict_class is deprecated in recent tensorflow, so we use this np.argmax construction when predicting
    predictions = np.concatenate([predictions, np.argmax(model.predict(x[None, :, :, :], verbose=0), axis=-1)],axis=None)
    labels = np.concatenate([labels, y.numpy()], axis=None)

print(tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())

pass