import tkinter as tk
from tkinter import filedialog
from tkinter import Entry
from PIL import Image, ImageTk
import numpy as np
from kmeans import KMeanClustering
import cv2

class ImageCompressorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compressor App")
        self.image_path = ""
        self.k = 4
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()
        self.k_label = tk.Label(root, text="Enter Cluster:")
        self.k_label.pack()
        self.k_entry = Entry(root)
        self.k_entry.insert(0, str(self.k))
        self.k_entry.pack()

        self.compress_button = tk.Button(root, text="Compress", command=self.compress_image)
        self.compress_button.pack()

        self.compressed_label = tk.Label(root)
        self.compressed_label.pack()

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if self.image_path:
            img = Image.open(self.image_path)
            self.photo = ImageTk.PhotoImage(img)
            self.compressed_label.config(image=self.photo)
            self.compressed_label.image = self.photo

    def compress_image(self):
        if self.image_path:
            img = read_image(self.image_path)
            self.k = int(self.k_entry.get())
            compressed_img = compress_image(img, self.k)
            compressed_img = Image.fromarray((compressed_img * 255).astype('uint8'))
            self.photo = ImageTk.PhotoImage(compressed_img)
            self.compressed_label.config(image=self.photo)
            self.compressed_label.image = self.photo

def read_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

def compress_image(img, k):
    img_reshaped = img.reshape(-1, 3)
    label, centroids = KMeanClustering(k=k).fit(img_reshaped)
    compressed_img = centroids[label[-1]].reshape(img.shape)
    return compressed_img

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressorApp(root)
    root.mainloop()
