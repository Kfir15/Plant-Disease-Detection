# Import the necessary libraries
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


model = tf.keras.models.load_model('D:\Deep Learning\FinalProject\PlantDiseaseModel.h5', compile = False)


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

