#!/usr/bin/env python3
"""
Creates a minimalist visualization for the Simulation-to-Real Transfer section.
Focuses on clean, intuitive visual with minimal text.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrowPatch, PathPatch
import matplotlib.path as mpath
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'images')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_minimalist_reality_gap():
    """
    Creates a clean, minimalist visualization of the reality gap concept.
    Uses simple shapes, bold colors, and minimal text.
    """
    print("Creating minimalist reality gap visualization...")
    
    # Create a large figure with a clean white background
    plt.figure(figsize=(16, 8), facecolor='white')
    
    # Set up the plot area with equal aspect ratio
    ax = plt.axes([0.01, 0.01, 0.98, 0.98])
    ax.set_aspect('equal')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # ---- SIMULATION SIDE (LEFT) ----
    # Create a clean blue background for simulation
    sim_bg = Rectangle((0.5, 0.5), 6, 7, facecolor='#E1F5FE', linewidth=0)
    ax.add_patch(sim_bg)
    
    # Simple robot arm in simulation - perfect, geometric shapes
    # Base
    ax.add_patch(Rectangle((1.5, 1.0), 1.5, 0.8, facecolor='#90A4AE', edgecolor='#546E7A', linewidth=2))
    # Arm parts (uniform, perfect geometry)
    ax.add_patch(Rectangle((2.0, 1.8), 0.5, 2.0, facecolor='#2196F3', edgecolor='#1565C0', linewidth=2))
    ax.add_patch(Rectangle((1.5, 3.8), 1.5, 0.5, facecolor='#2196F3', edgecolor='#1565C0', linewidth=2))
    # End effector
    ax.add_patch(Circle((1.5, 4.3), 0.4, facecolor='#F44336', edgecolor='#C62828', linewidth=2))
    
    # Target object in simulation - perfect square
    ax.add_patch(Rectangle((4.5, 3.0), 1.0, 1.0, facecolor='#FFB300', edgecolor='#FB8C00', linewidth=2))
    
    # Ground - clean, straight line
    ax.add_patch(Rectangle((1, 0.8), 5, 0.2, facecolor='#ECEFF1', edgecolor='#B0BEC5', linewidth=2))
    
    # "Simulation" label at the top (minimal text)
    ax.text(3.5, 7.2, "SIMULATION", ha='center', fontsize=14, fontweight='bold', color='#1976D2')
    
    # ---- REALITY SIDE (RIGHT) ----
    # Create a subtle beige background for reality
    real_bg = Rectangle((9.5, 0.5), 6, 7, facecolor='#FFF8E1', linewidth=0)
    ax.add_patch(real_bg)
    
    # Real robot with imperfections - less perfect shapes, slight deviations
    # Base - slightly asymmetrical
    real_base = Polygon([(10.5, 1.0), (12.1, 1.05), (12.0, 1.8), (10.45, 1.75)], 
                        facecolor='#78909C', edgecolor='#455A64', linewidth=2)
    ax.add_patch(real_base)
    
    # Arm parts (slight wear, imperfections)
    # Main arm - slightly bent
    path = mpath.Path([
        (11.0, 1.8), (11.5, 1.85), (11.55, 3.7), (11.1, 3.8)
    ])
    arm1 = PathPatch(path, facecolor='#1E88E5', edgecolor='#0D47A1', linewidth=2)
    ax.add_patch(arm1)
    
    # Horizontal arm piece
    ax.add_patch(Polygon([(10.4, 3.8), (11.9, 3.85), (11.85, 4.3), (10.5, 4.25)], 
                        facecolor='#1E88E5', edgecolor='#0D47A1', linewidth=2))
    
    # End effector - slightly worn
    ax.add_patch(Circle((10.5, 4.3), 0.4, facecolor='#E53935', edgecolor='#B71C1C', linewidth=2))
    
    # Target object in reality - slightly aged, not perfect
    ax.add_patch(Polygon([(13.5, 3.0), (14.55, 3.05), (14.5, 4.1), (13.45, 4.0)], 
                        facecolor='#FFA000', edgecolor='#F57C00', linewidth=2))
    
    # Ground - slightly uneven
    ground_points = [(10, 0.8)]
    rng = np.random.RandomState(42)
    for x in np.linspace(10, 15, 10):
        ground_points.append((x, 0.8 + rng.uniform(-0.05, 0.05)))
    ground_points.append((15, 0.8))
    ground_points.append((15, 0.6))
    ground_points.append((10, 0.6))
    
    ground = Polygon(ground_points, facecolor='#E0E0E0', edgecolor='#9E9E9E', linewidth=2)
    ax.add_patch(ground)
    
    # "Reality" label at the top (minimal text)
    ax.text(12.5, 7.2, "REALITY", ha='center', fontsize=14, fontweight='bold', color='#FF8F00')
    
    # ---- THE GAP BETWEEN SIMULATION AND REALITY ----
    # Central gap area with key discrepancy indicators
    gap_bg = Polygon([(7, 0.5), (9, 0.5), (9, 7.5), (7, 7.5)], 
                    facecolor='#FAFAFA', edgecolor='#E0E0E0', linewidth=2, alpha=0.8)
    ax.add_patch(gap_bg)
    
    # Diverging arrows showing the gap
    arrow_props = dict(arrowstyle='->', lw=3, joinstyle='miter')
    sim_arrow = FancyArrowPatch((6.6, 4), (7.3, 4.8), 
                               color='#1976D2', connectionstyle="arc3,rad=-0.3", **arrow_props)
    real_arrow = FancyArrowPatch((9.4, 4), (8.7, 4.8), 
                                color='#FF8F00', connectionstyle="arc3,rad=0.3", **arrow_props)
    
    ax.add_patch(sim_arrow)
    ax.add_patch(real_arrow)
    
    # Gap zones represented by simple icons (no text)
    # Physics gap (springs/dynamics)
    spring = Polygon([
        (8, 6), (8, 5.8), (7.7, 5.7), (8.3, 5.5), (7.7, 5.3), (8.3, 5.1), (7.7, 4.9), (8, 4.8), (8, 4.6),
    ], edgecolor='#7B1FA2', facecolor='none', linewidth=3)
    ax.add_patch(spring)
    
    # Sensor gap (simplified eye/camera)
    sensor = Circle((8, 3.5), 0.4, facecolor='none', edgecolor='#00897B', linewidth=3)
    pupil = Circle((8, 3.5), 0.15, facecolor='#00897B', edgecolor='none')
    ax.add_patch(sensor)
    ax.add_patch(pupil)
    
    # Appearance gap (simplified graphics)
    ax.add_patch(Rectangle((7.6, 2.1), 0.35, 0.35, facecolor='#1976D2', linewidth=0))
    ax.add_patch(Polygon([(8.05, 2.1), (8.4, 2.2), (8.35, 2.5), (8.05, 2.45)], 
                        facecolor='#FF8F00', linewidth=0))
    
    # Minimal label for the gap
    ax.text(8, 0.9, "GAP", ha='center', fontsize=14, fontweight='bold', color='#424242')
    
    # Save the figure
    plt.tight_layout(pad=0.5)
    output_path = os.path.join(output_dir, 'reality_gap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Minimalist reality gap visualization saved to {output_path}")

if __name__ == "__main__":
    create_minimalist_reality_gap()
    print("Minimalist sim-to-real visualization complete!") 