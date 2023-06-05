
# Import the Tkinter library
from tkinter import Tk, Button, Label, StringVar, Entry, IntVar, END, W, E
from model import load_model
import object_detection





def main():
    net, output_layers = load_model('yolov3-tiny.weights', 'yolov3-tiny.cfg')
    print("Model loaded successfully.")

# Create a new window
root = Tk()

# Set the title of the window
root.title("Preschool Mathematics Education Based on Image Detection")

# Set the window background color
root.configure(bg='lightblue')

# Create a label for the program name
label = Label(root, text="Preschool Mathematics Education Based on Image Detection", bg='lightblue', fg='black', font=('helvetica', 24, 'bold'))
label.grid(row=0, column=0, padx=30, pady=30)

# Create a new button
button1 = Button(root, text="Item Quantity Detection", bg='white', fg='black', font=('helvetica', 16, 'bold'), width=30, height=2, command=object_detection.open_detection_window)

# Add the button to the window using grid
button1.grid(row=1, column=0, padx=10, pady=10)

# Create other buttons
button2 = Button(root, text="Handwritten Digit Detection", bg='white', fg='black', font=('helvetica', 16, 'bold'), width=30, height=2)
button2.grid(row=2, column=0, padx=10, pady=10)

button3 = Button(root, text="Arithmetic Symbol Detection", bg='white', fg='black', font=('helvetica', 16, 'bold'), width=30, height=2)
button3.grid(row=3, column=0, padx=10, pady=10)

button4 = Button(root, text="Equation Recognition Calculation", bg='white', fg='black', font=('helvetica', 16, 'bold'), width=30, height=2)
button4.grid(row=4, column=0, padx=10, pady=10)

button5 = Button(root, text="Handwritten Calculation Result Detection", bg='white', fg='black', font=('helvetica', 16, 'bold'), width=30, height=2)
button5.grid(row=5, column=0, padx=10, pady=10)

# Run the main function
main()

# Run the window
root.mainloop()

