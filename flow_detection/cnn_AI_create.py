# TensorFlow and tf.keras
import tensorflow as tf

#from tensorflow import keras

#import keras
# from keras.layers import Dense
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import load_model
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from keras.optimizers import Adam

#https://www.tensorflow.org/tutorials/images/classification

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#                                                     NEED TO TYPE!!!!!!!!!!!!!!!!!!
print("Hello World")
# Please write the path to the ‘supervisor’ file.
data_dir = r"C:\Users\josep\OneDrive - UCB-O365\Students\_shares\Lee HUB\junresearch\DeeplearningCNN_flow_detection\supervisor"

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


# Create ImageDataGenerator to load and preprocess image data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,  # image normalization
    validation_split=0.2  # Set verification data split ratio
)

# Load data for training
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(100, 100),  # Image resizing (Jun had 64x64)
    batch_size=4,          # Set batch size (Jun had 64)
    class_mode='categorical',  # Multi-class classification
    subset='training'          # training data
)

# Load data for validation
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(100, 100),  # Jun had 64x64
    batch_size=4, #Jun had 64
    class_mode='categorical',
    subset='validation'  # validation data
)



#Find an appropriate avtivation instead of 'relu' to reduce loss
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes, this number is the number of files in 'supervisor'
])




# Compile the model with the desired learning rate
# Jun had learning_rate at 0.0001
#new_learning_rate = 0.0001
#optimizer = tf.keras.optimizers.Adam(learning_rate=new_learning_rate)

optimizer = tf.keras.optimizers.Adam(gradient_accumulation_steps=2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set class weights. By doing this, the difference in the number of photos in each file is overcome.
#class_weight = {"flow": 0.50, "noflow": 0.20, "snow": 0.30}

# adding early stopping (https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_03_4_early_stop.ipynb)
monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=1e-3,
                                           patience=5,
                                           mode='auto',
                                           restore_best_weights=True)

    # model training
history = model.fit(
        train_generator,
        epochs=15,  # Set epochs count
        validation_data=validation_generator,
    callbacks=[monitor]
    )

    # Model evaluation and self-diagnosis
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {test_accuracy}')

    # Image classification prediction
predictions = model.predict(validation_generator)



#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#                                                     NEED TO TYPE!!!!!!!!!!!!!!!!!!

# Specify the path and file name to save the model C:/path/file_name.h5
model.save(r"C:\Users\josep\Documents\GitHub\MLtests\flow_detection\test_model.keras")

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################