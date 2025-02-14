from PIL import Image
import random
import matplotlib.pyplot as plt
import os as os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Dense, Dropout, Flatten, Input, BatchNormalization, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------------------------------------------
# CONNECTING DATA
print("---CONNECTING DATA---") # connect the DATA

# Set the directories for training and testing images
train_healthy_dir = 'data/Train/Healthy'
train_powdery_dir = 'data/Train/Powdery'
train_rust_dir = 'data/Train/Rust'

test_healthy_dir = 'data/Test/Healthy'
test_powdery_dir = 'data/Test/Powdery'
test_rust_dir = 'data/Test/Rust'

# Set the path for training, test, and real directories
train_path = 'data/Train'
test_path = 'data/Test'
real_path = 'data'

print("---finished to connect---")
print("")

# -------------------------------------------------------------------
# FILE PATHS
# Print the file paths to verify the loaded directories
print("files path: ")
for dirname, _, filenames in os.walk(real_path):
    print(dirname)
print("")

# -------------------------------------------------------------------
# SHUFFLE FILE NAMES
# Shuffle the file names in the training and testing directories for each category

train_healthy_names = os.listdir(train_healthy_dir)
random.shuffle(train_healthy_names)
print(train_healthy_names[:10])

train_powdery_names = os.listdir(train_powdery_dir)
random.shuffle(train_powdery_names)
print(train_powdery_names[:10])

train_rust_names = os.listdir(train_rust_dir)
random.shuffle(train_rust_names)
print(train_rust_names[:10])


test_healthy_hames = os.listdir(test_healthy_dir)
print(test_healthy_hames[:10])

test_powdery_names = os.listdir(test_powdery_dir)
print(test_powdery_names[:10])

test_rust_names = os.listdir(test_rust_dir)
print(test_rust_names[:10])


real_names = os.listdir(real_path)
print(real_names)

print("")

# -------------------------------------------------------------------
# RESIZE IMAGES
# Resize the images to a size of 150x150 pixels

print("start to change the size of the picture", end='s')

small = (150, 150)
category_names = os.listdir(train_path)
category_names = os.listdir(train_path)

# Resize training images
for category in category_names:
    category_path = os.path.join(train_path, category)
    for file in os.listdir(category_path):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(category_path, file)
            with Image.open(img_path) as img:
                img.thumbnail(small)
                img.save(img_path)

# Resize testing images
category_names = os.listdir(test_path)
for category in category_names:
    category_path = os.path.join(test_path, category)
    for file in os.listdir(category_path):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(category_path, file)
            with Image.open(img_path) as img:
                img.thumbnail(small)
                img.save(img_path)


print("-----> Done;")
print("")

# -------------------------------------------------------------------

# DATA GENERATION
# Create the ImageDataGenerator for training and testing data
train_datagen=ImageDataGenerator(rescale=1/255)
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size= (150,150),
    batch_size = 48,
    class_mode = 'categorical')

test_datagen=ImageDataGenerator(rescale=1/255)
test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size = 48,
    class_mode = 'categorical')

# -------------------------------------------------------------------
# MODEL ARCHITECTURE
# Create the model architecture

print("---START TO BUILD THE MODEL---")
print("")

# Create the model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten()) # Flatten the output tensor from the convolutional layers

# Add dense layers for classification
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=3, activation='softmax'))


# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

print("")

# -------------------------------------------------------------------
# MODEL TRAINING
# Train the model
print("---START TRAIN THE MODEL---")
history = model.fit(training_set,
                    validation_data=test_set,
                    epochs=10,
                    steps_per_epoch=len(training_set),
                    validation_steps=len(test_set))
print("")

# -------------------------------------------------------------------
# MODEL EVALUATION
# Evaluate the model on the test set
model.evaluate(test_set)

#model.save("PlantDiseaseModel.h5")
# -------------------------------------------------------------------
# PLOT ACCURACY
#accurracy VS validation accuracy

# Get the accuracy and validation accuracy values from the history object
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Set the figure size and font style
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 14})

# Plot the accuracy and validation accuracy curves
plt.plot(accuracy, linewidth=2, color='red')
plt.plot(val_accuracy, linewidth=2, color='green')

# Set the plot title, labels, and legend
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

# Add grid lines and tighten the layout
plt.grid(True)
plt.tight_layout()

# Show the plot
print("open graph - Model Accurac", end='y')
plt.show()
print("-------> close;")
print("")

# -------------------------------------------------------------------
# PLOT LOSS
#loss VS validation loss

# Get the loss and validation loss values from the history object
loss = history.history['loss']
val_loss = history.history['val_loss']

# Set the figure size and font style
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 14})

# Plot the loss and validation loss curves
plt.plot(loss, linewidth=2, color='red')
plt.plot(val_loss, linewidth=2, color='green')

# Set the plot title, labels, and legend
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

# Add grid lines and tighten the layout
plt.grid(True)
plt.tight_layout()

# Show the plot
print("open graph - Model Los", end='s')
plt.show()
print("-------> close;")
print("")

# -------------------------------------------------------------------
# PREDICTION FUNCTION
def output(image_path, class_indices):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Use the model to predict the class probabilities of the input image
    class_probabilities = model.predict(img_array)[0]

    # Convert the class probabilities to a class label
    predicted_class_index = np.argmax(class_probabilities)
    predicted_class = class_indices[predicted_class_index]

    print('Predicted class:', predicted_class)
    return predicted_class

# Define the class indices for prediction
class_indices = {0: 'healthy', 1: 'powdery', 2: 'rust'}


# USER INTERFACE
# Create a user interface to browse and predict images

# Import the necessary libraries
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

# -------------------------------------------------------------------
# Function to browse and display an image
def browse_image():
    global img
    global canvas
    global prediction_label

    # Open a file dialog to select an image file
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                          filetypes=(("Image Files", "*.jpg *.jpeg *.png"),))
    if filename:
        # Open and resize the selected image
        img = Image.open(filename)
        img = img.resize((500, 500))
        img = ImageTk.PhotoImage(img)

        # Update the canvas with the new image
        canvas.image = img
        canvas.create_image(0, 0, anchor="nw", image=img)

        # Make a prediction using the selected image
        prediction_label.config(text="Prediction: " + output(filename, class_indices))


# Function to browse and display an image
def browse_image():
    global img
    global canvas
    global prediction_label

    # Open a file dialog to select an image file
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                          filetypes=(("Image Files", "*.jpg *.jpeg *.png"),))
    if filename:
        # Open and resize the selected image
        img = Image.open(filename)
        img = img.resize((500, 500))
        img = ImageTk.PhotoImage(img)

        # Update the canvas with the new image
        canvas.image = img
        canvas.create_image(0, 0, anchor="nw", image=img)

        # Make a prediction using the selected image
        prediction_label.config(text="Prediction: " + output(filename, class_indices))


# Function to reset the image and prediction label
def reset_image():
    canvas.delete("all")
    prediction_label.config(text="Prediction: ")


# Function to exit the prediction window
def exit_prediction_window():
    prediction_window.destroy()


# Function to exit the start window
def exit_start_window():
    start_window.destroy()


# Function to open the prediction window and start the application
def open_prediction_window():
    global prediction_window
    global canvas
    global prediction_label
    global img

    # Close the start window
    start_window.destroy()

    # Create the prediction window
    prediction_window = tk.Tk()
    prediction_window.geometry("700x750")
    prediction_window.title("Plant Disease Recognition - Prediction Window")

    # Label to prompt the user to upload an image
    lblNum = tk.Label(prediction_window, text="Please upload a picture:", height=2, font=("Arial", 24))
    lblNum.pack(side="top")

    # Button to browse and upload an image
    browse_button = tk.Button(prediction_window, text="Upload", command=browse_image, width=10, font=("Arial", 15))
    browse_button.pack(pady=10)

    # Button to reset the image and prediction
    reset_button = tk.Button(prediction_window, text="Reset", command=reset_image, width=10, fg='red',
                             font=("Arial", 15))
    reset_button.pack(pady=10)

    # Canvas to display the uploaded image
    canvas = tk.Canvas(prediction_window, width=500, height=500)
    canvas.pack()

    # Label to display the prediction result
    prediction_label = tk.Label(prediction_window, text="Prediction: ", font=("Arial", 15))
    prediction_label.pack()

    # Button to exit the prediction window
    exit_button = tk.Button(prediction_window, text="Exit", command=exit_prediction_window, width=10,
                            font=("Arial", 15))
    exit_button.place(x=20, y=700)

    # Run the prediction window
    prediction_window.mainloop()


# Create the start window
start_window = tk.Tk()
start_window.geometry("600x300")
start_window.title("Plant Disease Recognition - Start Window")

# Label to welcome the user
lblNum = tk.Label(start_window, text="Welcome to Plant Disease Recognition", height=2, font=("Arial", 24))
lblNum.pack(side="top")

# Button to start the application
start_button = tk.Button(start_window, text="Start", command=open_prediction_window, width=20, font=("Arial", 18))
start_button.pack(pady=50)

# Button to exit the application
exit_button = tk.Button(start_window, text="Exit", command=exit_start_window, width=20, font=("Arial", 18))
exit_button.pack(pady=10)

# Configure the start window to center the widgets
start_window.grid_rowconfigure(0, weight=1)
start_window.grid_rowconfigure(2, weight=1)
start_window.grid_columnconfigure(0, weight=1)

# Run the start window
start_window.mainloop()

