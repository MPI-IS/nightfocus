"""
Example script demonstrating camera focus optimization.

This script shows how to use the Camera abstraction and optimize_focus function
to automatically find the best focus position for a simulated camera.
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import nightfocus
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nightfocus import SimulatedCamera, optimize_focus
from nightfocus.focus_metrics import FOCUS_MEASURES

def plot_focus_history(history, output_file=None):
    """Plot the focus optimization history."""
    focus_values = [x[0] for x in history]
    scores = [x[1] for x in history]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(focus_values, scores, c='b', alpha=0.6, label='Measurements')
    
    # Sort for the line plot
    sorted_indices = np.argsort(focus_values)
    plt.plot(
        np.array(focus_values)[sorted_indices],
        np.array(scores)[sorted_indices],
        'r--',
        alpha=0.5,
        label='Interpolation'
    )
    
    # Mark the best focus
    best_idx = np.argmax(scores)
    plt.scatter(
        focus_values[best_idx],
        scores[best_idx],
        c='r',
        s=100,
        marker='*',
        label=f'Best Focus ({focus_values[best_idx]:.1f})'
    )
    
    plt.xlabel('Focus Position')
    plt.ylabel('Focus Score')
    plt.title('Focus Optimization History')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    # Create a simulated camera
    camera = SimulatedCamera(image_shape=(200, 200), noise_level=0.1)
    
    print("Optimizing focus using Bayesian optimization...")
    print(f"Using focus measure: tenengrad")
    
    # Run the optimization
    best_focus, history = optimize_focus(
        camera=camera,
        focus_measure=FOCUS_MEASURES["tenengrad"],
        bounds=(0, 100),
        initial_points=5,
        max_iter=15,
        random_state=42
    )
    
    print(f"\nOptimization complete!")
    print(f"Best focus position: {best_focus}")
    print(f"Number of evaluations: {len(history)}")
    
    # Plot the results
    plot_focus_history(history, "focus_optimization.png")
    
    # Show the best image
    best_img = camera.take_picture(int(best_focus))
    plt.figure(figsize=(8, 8))
    plt.imshow(best_img, cmap='gray')
    plt.title(f'Best Focus (Position: {best_focus})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("best_focus_image.png", dpi=150, bbox_inches='tight')
    print("Best focus image saved to best_focus_image.png")
    plt.show()

if __name__ == "__main__":
    main()
