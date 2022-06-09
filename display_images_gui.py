import os
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image

def display_images(critical_images):
    root = Tk()
    root.title("Critical Images")
    root.geometry("500x400")

    # Create Main Frame
    main_frame = Frame(root)
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
    canvas.create_window((0,0), window=second_frame, anchor="nw")

    # Generate Rows
    image_objects = []
    for element in critical_images:
        image = Image.open(element.path).resize((60, 60))
        image_objects.append(ImageTk.PhotoImage(image))

    for i in range(len(critical_images)):
        Label(second_frame, image=image_objects[i]).grid(row=i, column=0, pady=10, padx=10)
        #Button(second_frame, text="Buttontext").grid(row=i, column=1, pady=10, padx=10)
        label = "Resolution: " + str(critical_images[i].shape[0]) + " x " + str(critical_images[i].shape[1]) + " px\n" + "Probability: " + str(round(critical_images[i].probability, 2)) + " %"
        Label(second_frame, text=label, anchor="e").grid(row=i, column=1, pady=10, padx=10)

    root.mainloop()