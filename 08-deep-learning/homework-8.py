import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import requests
import zipfile
import os

url = 'https://github.com/SVizor42/ML_Zoomcamp/releases/download/bee-wasp-data/data.zip'

local_zip_file = 'data.zip'

response = requests.get(url)
with open(local_zip_file, 'wb') as file:
    file.write(response.content)

if os.path.exists(local_zip_file):
    print("File downloaded successfully.")
else:
    print("File download failed.")

with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
    zip_ref.extractall()

print("Extraction complete.")


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.002, momentum=0.8)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()


# Create data generators for train and test sets
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Define batch size and directories for train and test data
batch_size = 20
train_dir = 'data/train/'
test_dir = 'data/test/'

# Create generators with flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',  # Binary classification
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',  # Binary classification
    shuffle=True
)

# Now, you can use model.fit() with these generators
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Extract training accuracy from the training history
training_accuracies = history.history['accuracy']

# Calculate the median training accuracy
median_training_accuracy = np.median(training_accuracies)

print("Median Training Accuracy:", median_training_accuracy)

# Extract training loss from the training history
training_losses = history.history['loss']

# Calculate the standard deviation of training loss
std_training_loss = np.std(training_losses)

print("Standard Deviation of Training Loss:", std_training_loss)

# Create data generator for training data with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define batch size and directory for train data
batch_size = 20

# Create a generator with flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary',  # Binary classification
    shuffle=True
)

# Assuming you have already defined and compiled your model
# train_generator and test_generator are already defined with data augmentation

# Continue training the model for 10 more epochs
history_augmented = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Calculate the mean of test loss for all the epochs including the previous ones
all_test_losses = history.history['val_loss'] + history_augmented.history['val_loss']
mean_test_loss = np.mean(all_test_losses)

print("Mean Test Loss (including augmented epochs):", mean_test_loss)

# Extract test accuracy from the augmented training history for the last 5 epochs (epochs 6 to 10)
test_accuracy_last_5_epochs = history_augmented.history['val_accuracy'][5:]

# Calculate the average test accuracy for the last 5 epochs
average_test_accuracy_last_5_epochs = np.mean(test_accuracy_last_5_epochs)

print("Average Test Accuracy (last 5 epochs):", average_test_accuracy_last_5_epochs)

