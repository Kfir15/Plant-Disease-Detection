import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image


def browse_image():
    global img
    global canvas
    global prediction_label

    filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                          filetypes=(("Image Files", "*.jpg *.jpeg *.png"),))
    if filename:
        img = Image.open(filename)
        img = img.resize((500, 500))  # Adjust the size of the displayed image
        img = ImageTk.PhotoImage(img)
        canvas.image = img  # Save reference to keep the image object alive
        canvas.create_image(0, 0, anchor="nw", image=img)
        prediction_label.config(text="Prediction: ")


def reset_image():
    canvas.delete("all")
    prediction_label.config(text="Prediction: ")


def exit_prediction_window():
    prediction_window.destroy()


def open_prediction_window():
    global prediction_window
    global canvas
    global prediction_label
    global img

    start_window.destroy()  # Close the start window
    prediction_window = tk.Tk()
    prediction_window.geometry("750x800")  # determines the size of the window
    prediction_window.title("Plant Disease Recognition - Kfir Eitan")  # add title to the window

    lblNum = tk.Label(prediction_window, text="Please upload picture:", height=2, font=("Arial", 24))
    lblNum.pack(side="top")

    browse_button = tk.Button(prediction_window, text="Upload", command=browse_image, width=10, font=("Arial", 15))
    browse_button.pack(pady=10)  # Increase the vertical spacing

    reset_button = tk.Button(prediction_window, text="Reset", command=reset_image, width=10, fg='red',
                             font=("Arial", 15))
    reset_button.pack(pady=10)  # Increase the vertical spacing

    canvas = tk.Canvas(prediction_window, width=500, height=500)  # Adjust the canvas size
    canvas.pack()

    prediction_label = tk.Label(prediction_window, text="Prediction: ", font=("Arial", 15))
    prediction_label.pack()

    exit_button = tk.Button(prediction_window, text="Exit", command=exit_prediction_window, width=10,
                            font=("Arial", 15))
    exit_button.place(x=20, y=750)  # Position the button at the bottom left corner

    prediction_window.mainloop()


start_window = tk.Tk()
start_window.geometry("600x300")
start_window.title("Plant Disease Recognition - Kfir Eitan - Start Window")

lblNum = tk.Label(start_window, text="Welcome to plant disease recognition", height=2, font=("Arial", 24))
lblNum.pack(side="top")

start_button = tk.Button(start_window, text="Start", command=open_prediction_window, width=10, font=("Arial", 15))
start_button.pack(pady=50)

start_window.mainloop()
