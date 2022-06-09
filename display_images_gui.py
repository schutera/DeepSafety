from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image

root = Tk()
root.title("Critical Images")
root.geometry("500x400")

image_path = "./safetyBatches/Batch_1/42/00000_00000.jpg"

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

for i in range(100):
    Button(second_frame, text="Buttontext").grid(row=i, column=0, pady=10, padx=10)

my_image = ImageTk.PhotoImage(Image.open(image_path))
my_label = Label(second_frame, text="itsfriday").grid(row=3,column=2)

label = Label(second_frame, image=my_image).grid(row=4, column=2)

root.mainloop()