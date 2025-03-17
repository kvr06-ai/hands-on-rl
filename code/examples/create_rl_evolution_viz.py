#!/usr/bin/env python3
"""
Creates a visualization showing the evolution of reinforcement learning environments
from classic control problems to modern simulation platforms.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
from matplotlib.colors import LinearSegmentedColormap
import os

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_rl_evolution_timeline():
    """
    Creates a visual timeline of RL environment evolution, highlighting key milestones.
    """
    print("Creating RL environment evolution visualization...")
    
    # Create figure and axis with appropriate size
    fig, ax = plt.figure(figsize=(14, 8)), plt.gca()
    plt.subplots_adjust(left=0.07, right=0.93, top=0.9, bottom=0.1)
    
    # Background styling
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(1980, 2025)
    ax.set_ylim(0, 10)
    ax.set_yticks([])
    
    # Create timeline backbone
    ax.plot([1980, 2025], [5, 5], 'k-', linewidth=2, alpha=0.7)
    
    # Define key era colors with a progression
    era_colors = {
        "Classic Control": "#5D8AA8",       # Cool blue
        "Arcade Learning": "#8B72BE",       # Purple
        "Physical Simulation": "#33A02C",   # Green
        "Multi-Agent": "#FF7F00",           # Orange
        "Procedural Generation": "#D62728", # Red
        "GPU-Accelerated": "#E31A1C"        # Bright red
    }
    
    # Define eras with start years, end years, vertical positions
    eras = [
        {"name": "Classic Control", "start": 1983, "end": 2010, "y": 7, 
         "description": "Low-dimensional state spaces,\nclear reward structures"},
        {"name": "Arcade Learning", "start": 2013, "end": 2022, "y": 3, 
         "description": "Learning from pixels,\nhigh-dimensional observations"},
        {"name": "Physical Simulation", "start": 2015, "end": 2023, "y": 8, 
         "description": "Continuous control,\nrealistic physics"},
        {"name": "Multi-Agent", "start": 2018, "end": 2025, "y": 2, 
         "description": "Emergent behaviors,\ncompetition & cooperation"},
        {"name": "Procedural Generation", "start": 2019, "end": 2025, "y": 9, 
         "description": "Domain randomization,\ngeneralization capabilities"},
        {"name": "GPU-Accelerated", "start": 2020, "end": 2025, "y": 1, 
         "description": "Massive parallelization,\nhigh-throughput training"}
    ]
    
    # Plot era regions
    for era in eras:
        # Create era regions as soft rectangles
        rect = patches.FancyBboxPatch(
            (era["start"], era["y"] - 0.8), 
            width=(era["end"] - era["start"]), 
            height=1.6,
            boxstyle=patches.BoxStyle("Round", pad=0.2, rounding_size=0.5),
            facecolor=era_colors[era["name"]], 
            alpha=0.15,
            edgecolor=era_colors[era["name"]], 
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Add era name and description
        ax.text(
            era["start"] + (era["end"] - era["start"])/2, 
            era["y"] + 0.5,
            era["name"], 
            fontsize=12, 
            fontweight='bold', 
            ha='center', 
            va='center', 
            color=era_colors[era["name"]]
        )
        ax.text(
            era["start"] + (era["end"] - era["start"])/2, 
            era["y"] - 0.3,
            era["description"], 
            fontsize=9, 
            ha='center', 
            va='center', 
            alpha=0.85,
            color='black'
        )
        
        # Connect to timeline
        ax.plot(
            [era["start"] + (era["end"] - era["start"])/2, era["start"] + (era["end"] - era["start"])/2], 
            [era["y"] - 0.8, 5], 
            ':', 
            color=era_colors[era["name"]], 
            alpha=0.5, 
            linewidth=1.5
        )
    
    # Add specific milestone environments
    milestones = [
        {"year": 1983, "name": "CartPole", "y": 6, "era": "Classic Control"},
        {"year": 1988, "name": "Mountain Car", "y": 7, "era": "Classic Control"},
        {"year": 1996, "name": "Acrobot", "y": 6, "era": "Classic Control"},
        {"year": 2013, "name": "Atari 2600 Games", "y": 4, "era": "Arcade Learning"},
        {"year": 2015, "name": "TORCS", "y": 3, "era": "Arcade Learning"},
        {"year": 2016, "name": "MuJoCo", "y": 8, "era": "Physical Simulation"},
        {"year": 2017, "name": "Roboschool/PyBullet", "y": 7, "era": "Physical Simulation"},
        {"year": 2018, "name": "PettingZoo", "y": 3, "era": "Multi-Agent"},
        {"year": 2018, "name": "Unity ML-Agents", "y": 9, "era": "Physical Simulation"},
        {"year": 2019, "name": "Procgen Benchmark", "y": 9, "era": "Procedural Generation"},
        {"year": 2019, "name": "Neural MMO", "y": 2, "era": "Multi-Agent"},
        {"year": 2020, "name": "Isaac Gym", "y": 1, "era": "GPU-Accelerated"},
        {"year": 2021, "name": "Meta-World", "y": 8, "era": "Procedural Generation"},
        {"year": 2022, "name": "XLand", "y": 2, "era": "Multi-Agent"},
        {"year": 2022, "name": "IsaacSim", "y": 2, "era": "GPU-Accelerated"}
    ]
    
    # Plot milestones as dots with annotations
    for milestone in milestones:
        ax.scatter(
            milestone["year"], 
            milestone["y"], 
            s=100, 
            color=era_colors[milestone["era"]], 
            zorder=10,
            edgecolor='white',
            linewidth=1
        )
        
        # Add milestone name with appropriate position
        if milestone["y"] > 5:
            va = 'bottom'
            offset = 0.3
        else:
            va = 'top'
            offset = -0.3
            
        ax.text(
            milestone["year"], 
            milestone["y"] + offset,
            milestone["name"], 
            fontsize=9, 
            ha='center', 
            va=va, 
            fontweight='normal', 
            zorder=10,
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1, boxstyle='round,pad=0.2')
        )
    
    # Add year ticks
    years = np.arange(1980, 2026, 5)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=10)
    ax.tick_params(axis='x', which='both', length=0, pad=5)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add title and source
    plt.title("Evolution of Reinforcement Learning Environments", fontsize=16, pad=20)
    plt.figtext(0.5, 0.01, "Note: Timeline shows approximate introduction dates of key environment paradigms and platforms", 
               ha="center", fontsize=10, style='italic')
    
    # Smooth out appearance
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, 'rl_environment_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"RL environment evolution visualization saved to {output_path}")

if __name__ == "__main__":
    create_rl_evolution_timeline()
    print("Visualization complete!") 