import os
import cv2
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model without compiling (since it is used only for inference)
model = tf.keras.models.load_model("facialemotionmodel.h5", compile=False)

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the image for the model
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Labels for emotion predictions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Initialize the Tkinter window
root = tk.Tk()
root.title("Facial Emotion Detection Dashboard")

# Create a frame to hold the entire dashboard (webcam, metrics, pie chart, and images)
main_frame = tk.Frame(root, bg='white')
main_frame.pack(fill=tk.BOTH, expand=True)

# Create four frames: one for the webcam, one for the metrics, one for the pie chart, and one for the images
video_frame = tk.Frame(main_frame, bg='white')
video_frame.grid(row=1, column=0, padx=0, pady=0)

metrics_frame = tk.Frame(main_frame, bg='white')
metrics_frame.grid(row=1, column=1, padx=100, pady=100)

chart_frame = tk.Frame(main_frame, bg='white')
chart_frame.grid(row=1, column=2, padx=0, pady=0)

image_frame = tk.Frame(main_frame, bg='black')  # New frame for the images
image_frame.grid(row=0, column=0, columnspan=3, padx=0, pady=10)  # Place it above the other frames

# Label for the Emotion Meter
emotion_meter_label = Label(metrics_frame, text="Emotion Count", font=("Helvetica", 20),bg='white')
emotion_meter_label.pack(pady=0)

# Initialize emotion counts
emotion_counts = {label: 0 for label in labels.values()}

# Create labels for each emotion's count
emotion_count_labels = {}
for emotion in emotion_counts:
    lbl = Label(metrics_frame, text=f"{emotion.capitalize()}: {emotion_counts[emotion]}", font=("Helvetica", 14), bg='white')
    lbl.pack(anchor='w', pady=2)
    emotion_count_labels[emotion] = lbl

# Create a figure for the pie chart
fig, ax = plt.subplots(figsize=(5, 5))
canvas = FigureCanvasTkAgg(fig, master=chart_frame)  # A tk.DrawingArea.
canvas.get_tk_widget().pack(padx=0, pady=0)

# Function to plot the pie chart
def plot_pie_chart():
    ax.clear()  # Clear the previous chart
    sizes = list(emotion_counts.values())
    labels_list = list(emotion_counts.keys())
    
    ax.pie(sizes, labels=labels_list, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie chart is a circle.
    ax.set_title("Emotion Distribution", fontsize=16, color='black')

    canvas.draw()  # Update the canvas with the new pie chart

# Label to show the video feed
video_label = Label(video_frame)
video_label.pack()

# Initialize webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables to track consecutive emotion predictions
current_emotion = None
consecutive_count = 0
threshold = 8  # Number of consecutive detections required to count the emotion

# Function to update the dashboard with the predicted emotion
def update_dashboard(prediction_label):
    # Update the current emotion display
    emotion_label = emotion_count_labels[prediction_label]
    emotion_label.config(text=f"{prediction_label.capitalize()}: {emotion_counts[prediction_label] + 1}")
    
    # Increment the emotion count only after 5 consecutive detections
    emotion_counts[prediction_label] += 1
    for emotion, count in emotion_counts.items():
        emotion_count_labels[emotion].config(text=f"{emotion.capitalize()}: {count}")
    
    # Update the pie chart
    plot_pie_chart()

# Function to capture video and perform emotion detection
def detect_and_display():
    global current_emotion, consecutive_count
    
    ret, im = webcam.read()
    if not ret:
        print("Error: Failed to capture image from webcam.")
        return

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_image = gray[y:y + h, x:x + w]
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Resize face for the model
        face_image = cv2.resize(face_image, (48, 48))
        img = extract_features(face_image)
        
        # Predict emotion
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        
        # Check if the same emotion is detected consecutively
        if prediction_label == current_emotion:
            consecutive_count += 1
        else:
            current_emotion = prediction_label
            consecutive_count = 1

        # If the same emotion is detected 5 times in a row, update the dashboard
        if consecutive_count == threshold:
            update_dashboard(prediction_label)
            consecutive_count = 0  # Reset the counter after counting the emotion
        
        # Display emotion on the video feed
        cv2.putText(im, prediction_label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert the image from OpenCV format to ImageTk format
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(im_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Schedule the next frame update
    video_label.after(10, detect_and_display)

# Function to release the webcam and close windows
def on_closing():
    webcam.release()  # Release the webcam
    cv2.destroyAllWindows()  # Destroy all OpenCV windows
    root.destroy()  # Close the Tkinter window

# Bind the closing event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Labels to display images side by side
#first_image_label = Label(image_frame)
#first_image_label.grid(row=0, column=0, padx=0, pady=0)  # First image on the left

second_image_label = Label(image_frame)
second_image_label.grid(row=0, column=1, padx=0, pady=0)  # Second image on the right

# Load an initial image for the first image label
#try:
    #initial_image = Image.open('')  # Load your image file
    #initial_image = initial_image.resize((900, 400))  # Resize the image if necessary
    #initial_image_tk = ImageTk.PhotoImage(initial_image)
    #first_image_label.configure(image=initial_image_tk,bg='white')
    #first_image_label.image = initial_image_tk  # Keep a reference to avoid garbage collection
#except Exception as e:
    #print(f"Error loading initial image: {e}")

# Load an initial image for the second image label
try:
    secondary_image = Image.open('E:\\Face emotion detection 1\\Face_Emotion_Recognition_Machine_Learning\\aid.png')  # Load your image file
    secondary_image = secondary_image.resize((900, 400))  # Resize the image if necessary
    secondary_image_tk = ImageTk.PhotoImage(secondary_image)
    second_image_label.configure(image=secondary_image_tk,bg='white')
    second_image_label.image = secondary_image_tk  # Keep a reference to avoid garbage collection
except Exception as e:
    print(f"Error loading secondary image: {e}")

# Start detecting and displaying
detect_and_display()

# Start the Tkinter main loop
root.mainloop()
