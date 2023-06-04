from tkinter import Toplevel, Label, Button, filedialog
import cv2
import numpy as np


from model import load_model

def count_objects(detections, class_names):
    # Initialize an empty dictionary to hold the counts
    counts = {}

    # Loop over the detections
    for detection in detections:
        # Get the class ID of the detection
        class_id = np.argmax(detection[5:])

        # Check if the class ID is within the range of the class names list
        if class_id < len(class_names):
            # Get the class name
            class_name = class_names[class_id]

            # If the class name is not in the counts dictionary, add it
            if class_name not in counts:
                counts[class_name] = 0

            # Increment the count of the class name
            counts[class_name] += 1

    # Return the counts dictionary
    return counts

def open_detection_window():
    # Create a new window
    detection_window = Toplevel()

    # Set the title of the window
    detection_window.title("Item Quantity Detection")

    # Create a label for the detection result
    label = Label(detection_window, text="Detection result will be displayed here.")
    label.pack()

    # Create a button for uploading an image
    upload_button = Button(detection_window, text="Upload Image", command=upload_image)
    upload_button.pack()

    # Create a button for returning to the main window
    return_button = Button(detection_window, text="Return to Main Window", command=detection_window.destroy)
    return_button.pack()

    # Load the YOLO model
    net, output_layers = load_model('yolov3-tiny.weights', 'yolov3-tiny.cfg')

    # Load the class names
    with open('coco.names', 'r') as f:
        class_names = f.read().splitlines()

    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to a blob
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Pass the blob through the network and get the detections
        net.setInput(blob)
        detections = net.forward(output_layers)

        # Get the counts of each object
        counts = count_objects(detections, class_names)

        # Update the label with the counts
        label.config(text=str(counts))

        # Show the frame
        cv2.imshow('Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Close the Tkinter window
    detection_window.destroy()


def upload_image():
    # Open a file dialog for the user to select an image
    filepath = filedialog.askopenfilename()

    # Load the image
    image = cv2.imread(filepath)

    # Convert the image to a blob
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Pass the blob through the network and get the detections
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Get the counts of each object
    counts = count_objects(detections, class_names)

    # Display the counts
    print(counts)