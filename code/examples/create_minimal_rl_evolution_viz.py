#!/usr/bin/env python3
"""
Creates a minimalist visualization showing the evolution of reinforcement learning environments.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_minimalist_rl_evolution_timeline():
    """
    Creates a minimalist visualization of the evolution of RL environments.
    The visualization focuses on visual storytelling with minimal text.
    """
    # Set up the figure with a clean, modern aesthetic
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    
    # Clean up the plot - remove spines and ticks
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=True, top=False, left=False, right=False)
    ax.set_yticks([])
    
    # Set the x-axis to represent years
    years = np.arange(1980, 2030, 5)
    ax.set_xlim(1980, 2025)
    ax.set_xticks(years)
    ax.set_xticklabels([str(year) for year in years], fontsize=11, color='#555555')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Timeline backbone
    ax.axhline(y=0.5, color='#AAAAAA', linewidth=1.5, alpha=0.7)
    
    # Define eras with subtle color bands
    era_colors = [
        [(1980, 1995), "#E6F2FF", "Classic Control"],  # Light blue
        [(1995, 2005), "#E6F5EC", "Board Games"],      # Light green
        [(2005, 2015), "#FFF2E6", "Physics Simulators"],  # Light orange
        [(2015, 2025), "#F5E6FF", "Modern Platforms"]   # Light purple
    ]
    
    for (start, end), color, era_name in era_colors:
        # Create subtle background color for the era
        rect = patches.Rectangle((start, 0.2), end-start, 0.6, 
                                 facecolor=color, alpha=0.7, edgecolor='none')
        ax.add_patch(rect)
        
        # Add era label at the bottom of each section
        ax.text((start + end)/2, 0.1, era_name, 
                ha='center', va='center', fontsize=10, color='#555555',
                fontweight='bold')
    
    # Define key milestones
    milestones = [
        (1983, 0.5, "Pole Balancing", 7, '#2E86C1'),        # Classic control
        (1992, 0.5, "BOXES", 6, '#2E86C1'),                 # Early RL algorithm
        (1997, 0.5, "TD-Gammon", 8, '#27AE60'),             # Board games
        (2004, 0.5, "AlphaGo", 10, '#27AE60'),              # Board games milestone
        (2008, 0.5, "RL Benchmark", 7, '#E67E22'),          # Early benchmark
        (2012, 0.5, "MuJoCo", 9, '#E67E22'),                # Physics sim
        (2016, 0.5, "OpenAI Gym", 10, '#8E44AD'),           # Framework
        (2019, 0.5, "Isaac Gym", 9, '#8E44AD'),             # Modern sim
        (2022, 0.5, "Embodied AI", 8, '#8E44AD')            # Latest trend
    ]
    
    # Add milestones as dots of varying sizes
    for year, y_pos, label, size, color in milestones:
        # Draw the milestone dot
        ax.scatter(year, y_pos, s=size*30, color=color, alpha=0.8, zorder=5)
        
        # Add minimal label with alternating positions
        if year % 10 < 5:  # Alternate above/below to avoid crowding
            y_text = y_pos + 0.15
            va = 'bottom'
        else:
            y_text = y_pos - 0.15
            va = 'top'
            
        # Use varying font weights based on importance
        if size >= 9:
            weight = 'bold'
        else:
            weight = 'normal'
            
        ax.text(year, y_text, label, ha='center', va=va, 
                fontsize=9, color='#333333', fontweight=weight)
    
    # Set the title in a modern, minimal style
    ax.text(1980, 1.05, 'Evolution of RL Environments', 
            fontsize=16, fontweight='bold', color='#333333')
    
    # Add a subtle caption
    ax.text(1980, 0.0, 'From simple control problems to GPU-accelerated simulations',
            fontsize=9, color='#777777', style='italic')
    
    plt.tight_layout()
    
    # Create the images directory if it doesn't exist
    os.makedirs('../../images', exist_ok=True)
    
    # Save the visualization
    plt.savefig('../../images/minimal_rl_evolution.png', dpi=300, bbox_inches='tight')
    print("Minimalist RL evolution visualization saved to '../../images/minimal_rl_evolution.png'")
    plt.close()

if __name__ == "__main__":
    # If running from the script's directory
    if os.path.basename(os.getcwd()) == 'examples':
        os.makedirs('../../images', exist_ok=True)
        output_path = '../../images/minimal_rl_evolution.png'
    # If running from the project root
    else:
        os.makedirs('images', exist_ok=True)
        output_path = 'images/minimal_rl_evolution.png'
    
    create_minimalist_rl_evolution_timeline()
    print("Visualization complete!") 