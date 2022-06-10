import os
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image


def display_images(critical_images):
    root = Tk()
    root.title("Critical Images")
    root.geometry("600x500")

    # Create Main Frame
    main_frame = Frame(root)
    Label(main_frame, text="Critical Images", anchor="nw", font=("Arial", 16)).pack(fill="both", pady=5, padx=10)
    Message(main_frame,
            text="The prediction of these images could be incorrect due to a low pixel resolution. "
                 "If the size of the sign in the image is too low, the pixel density could not suffice for predicting with a probability above 90 %.\n"
                 "Please check critical images on their prediction and consider expanding the training data on corresponding classes:",
            anchor="nw",
            font=("Arial", 10),
            aspect=700).pack(fill="both", padx=10, ipady=5)
    main_frame.pack(fill=BOTH, expand=1)

    # Create Canvas
    canvas = Canvas(main_frame)
    canvas.pack(side=LEFT, fill=BOTH, expand=1)

    # Add Scrollbar to Canvas
    scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
    scrollbar.pack(side=RIGHT, fill=Y)

    # Configure the Canvas
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Create Another Frame Inside the Canvas
    second_frame = Frame(canvas)

    # Add that new Frame to a Window in the Canvas
    canvas.create_window((0, 0), window=second_frame, anchor="nw")

    # Generate Rows
    image_objects = []
    for element in critical_images:
        image = Image.open(element.path).resize((70, 70))
        image_objects.append(ImageTk.PhotoImage(image))

    for i in range(len(critical_images)):
        Label(second_frame, image=image_objects[i]).grid(row=i, column=0, pady=10, padx=20)
        Message(second_frame,
                text="Filename:\nResolution:\nProbability:\nPrediction:\nGround Truth:",
                font=("Arial", 10),
                justify=LEFT,
                aspect=200).grid(row=i, column=1, pady=10)
        Message(second_frame,
                text=critical_images[i].file_name + "\n"
                     + str(critical_images[i].shape[0]) + " x " + str(critical_images[i].shape[1]) + " px\n"
                     + str(round(critical_images[i].probability, 2)) + " %\n"
                     + str(critical_images[i].prediction) + "\n"
                     + str(critical_images[i].ground_truth),
                font=("Arial", 10),
                justify=LEFT,
                aspect=200).grid(row=i, column=2, pady=10, padx=10)

    root.mainloop()
