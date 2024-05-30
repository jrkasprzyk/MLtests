from tensorflow import keras
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

#https://www.tensorflow.org/tutorials/images/classification

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#                                                     NEED TO TYPE!!!!!!!!!!!!!!!!!!

# Please write the path to the ‘supervisor’ file.
data_dir = "C:\Users\josep\OneDrive - UCB-O365\Students\_shares\Lee HUB\junresearch\DeeplearningCNN_flow_detection\supervisor"

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


# Create ImageDataGenerator to load and preprocess image data
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # image normalization
    validation_split=0.2  # Set verification data split ratio
)

# Load data for training
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),  # Image resizing
    batch_size=64,          # Set batch size
    class_mode='categorical',  # Multi-class classification
    subset='training'          # training data
)

# Load data for validation
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64), #Find an appropriate value to reduce loss
    batch_size=64, #Find an appropriate value to reduce loss
    class_mode='categorical',
    subset='validation'  # validation data
)



#Find an appropriate avtivation instead of 'relu' to reduce loss
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes, this number is the number of files in 'supervisor'
])




# Compile the model with the desired learning rate
new_learning_rate = 0.0001
optimizer = Adam(learning_rate=new_learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set class weights. By doing this, the difference in the number of photos in each file is overcome.
#class_weight = {"flow": 0.50, "noflow": 0.20, "snow": 0.30}

    # model training
history = model.fit(
        train_generator,
        epochs=15,  # Set epochs count
        validation_data=validation_generator,
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
model.save("C:/Users/junwo/Desktop/summer_research/cnn/jun_water_ditection.h5")

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################