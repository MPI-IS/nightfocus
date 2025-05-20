import tkinter as tk
from tkinter import ttk
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk


def create_focus_slideshow(dataset: Dict[int, np.ndarray], correct_focus: int):
    """
    Create and display a slideshow of images with varying focus values

    Args:
        dataset: Dictionary mapping focus values to image arrays
        correct_focus: The known correct focus value
    """
    # Sort focus values for sequential display
    sorted_focus = sorted(dataset.keys())
    current_index = 0

    # Create main window
    root = tk.Tk()
    root.title("Focus Slideshow Viewer")
    root.geometry("800x600")

    # Create matplotlib figure
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)

    # Frame for controls and labels
    control_frame = ttk.Frame(root)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    # Labels for focus information
    focus_label = ttk.Label(
        control_frame, text=f"Current Focus: {sorted_focus[current_index]}"
    )
    correct_label = ttk.Label(control_frame, text=f"Correct Focus: {correct_focus}")

    # Navigation buttons
    btn_frame = ttk.Frame(control_frame)
    prev_btn = ttk.Button(btn_frame, text="Previous", command=lambda: navigate(-1))
    next_btn = ttk.Button(btn_frame, text="Next", command=lambda: navigate(+1))

    # Layout widgets
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    focus_label.pack(side=tk.LEFT, padx=5)
    correct_label.pack(side=tk.LEFT, padx=5)
    btn_frame.pack(side=tk.RIGHT)
    prev_btn.pack(side=tk.LEFT, padx=2)
    next_btn.pack(side=tk.LEFT, padx=2)

    def update_display(index: int):
        """Update the displayed image and labels"""
        focus = sorted_focus[index]
        image_array = dataset[focus]

        ax.clear()
        ax.imshow(image_array, cmap="gray")
        ax.set_title(f"Focus Value: {focus}")
        canvas.draw()

        focus_label.config(text=f"Current Focus: {focus}")

    def navigate(delta: int):
        """Navigate through images"""
        nonlocal current_index
        new_index = current_index + delta
        if 0 <= new_index < len(sorted_focus):
            current_index = new_index
            update_display(current_index)

    # Initial display
    update_display(current_index)

    # Handle window closing
    def on_closing():
        plt.close("all")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the application
    root.mainloop()


# Example usage:
# create_focus_slideshow(dataset, config.correct_focus)
