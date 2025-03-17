#!/usr/bin/env python3
"""
This script creates an animated GIF illustrating the reinforcement learning cycle:
Agent → Action → Environment → State/Reward → Agent
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from matplotlib.path import Path
import matplotlib.patches as patches
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 7)
ax.axis('off')

# Create boxes for Agent and Environment
agent_box = Rectangle((1, 4), 3, 2, fill=True, facecolor='#3498db', alpha=0.8, edgecolor='black')
env_box = Rectangle((7, 4), 3, 2, fill=True, facecolor='#2ecc71', alpha=0.8, edgecolor='black')

# Add boxes to the plot
ax.add_patch(agent_box)
ax.add_patch(env_box)

# Add labels to the boxes
ax.text(2.5, 5, "Agent", ha='center', va='center', fontsize=14, color='white', fontweight='bold')
ax.text(8.5, 5, "Environment", ha='center', va='center', fontsize=14, color='white', fontweight='bold')

# Create arrows for the RL cycle
action_arrow = FancyArrowPatch((4, 5), (7, 5), arrowstyle='-|>', mutation_scale=20, linewidth=2, color='#e74c3c')
reward_arrow = FancyArrowPatch((7, 4.2), (4, 4.2), arrowstyle='-|>', mutation_scale=20, linewidth=2, color='#f39c12')

# Add arrows to the plot
ax.add_patch(action_arrow)
ax.add_patch(reward_arrow)

# Add labels to the arrows
ax.text(5.5, 5.3, "Action", ha='center', va='center', fontsize=12)
ax.text(5.5, 3.9, "State, Reward", ha='center', va='center', fontsize=12)

# Create a small circle to represent the learning cycle
learning_circle = Circle((5.5, 2), 0.1, fill=True, color='#e74c3c')
ax.add_patch(learning_circle)

# Create animation frames
frames = []

def update(frame):
    # Clear previous patches to avoid overlaps
    while len(ax.patches) > 4:  # Keep the two boxes and two arrows
        ax.patches[-1].remove()
    
    # Phase 1: Agent deciding
    if frame < 15:
        # Add thinking bubbles above agent
        thought_bubble = Circle((2.5 + 0.2*np.sin(frame/2), 6.3 + 0.1*np.cos(frame/2)), 
                               0.1 + 0.05*np.sin(frame/3), fill=True, color='#3498db', alpha=0.7)
        ax.add_patch(thought_bubble)
        
        # Smaller bubble
        small_bubble = Circle((2.8 + 0.1*np.sin(frame/1.5), 6.0 + 0.1*np.cos(frame/1.5)), 
                            0.07 + 0.03*np.sin(frame/2), fill=True, color='#3498db', alpha=0.5)
        ax.add_patch(small_bubble)
        
        # Highlight agent
        agent_highlight = Rectangle((1, 4), 3, 2, fill=False, edgecolor='#e74c3c', linewidth=3, 
                                   linestyle='-', alpha=0.5 + 0.5*np.sin(frame/3))
        ax.add_patch(agent_highlight)
    
    # Phase 2: Agent sends action
    elif frame < 30:
        # Move dot along action arrow
        pos = (frame - 15) / 15
        action_dot = Circle((4 + 3*pos, 5), 0.12, fill=True, color='#e74c3c')
        ax.add_patch(action_dot)
        
        # Highlight action arrow
        highlight_arrow = FancyArrowPatch((4, 5), (7, 5), arrowstyle='-|>', mutation_scale=20, 
                                        linewidth=3, color='#e74c3c', alpha=0.8)
        ax.add_patch(highlight_arrow)
    
    # Phase 3: Environment processes
    elif frame < 45:
        # Environment "processing" animation
        env_highlight = Rectangle((7, 4), 3, 2, fill=False, edgecolor='#27ae60', linewidth=3, 
                                linestyle='-', alpha=0.5 + 0.5*np.sin(frame/3))
        ax.add_patch(env_highlight)
        
        # Show some "calculation" inside environment
        calc_pos = np.sin((frame - 30) / 15 * np.pi * 2) * 0.3
        calc_dot1 = Circle((8.5 + calc_pos, 5 + calc_pos), 0.08, fill=True, color='#27ae60')
        calc_dot2 = Circle((8.5 - calc_pos, 5 - calc_pos), 0.08, fill=True, color='#27ae60')
        ax.add_patch(calc_dot1)
        ax.add_patch(calc_dot2)
    
    # Phase 4: Environment sends state/reward
    elif frame < 60:
        # Move dot along reward arrow
        pos = (frame - 45) / 15
        reward_dot = Circle((7 - 3*pos, 4.2), 0.12, fill=True, color='#f39c12')
        ax.add_patch(reward_dot)
        
        # Highlight reward arrow
        highlight_arrow = FancyArrowPatch((7, 4.2), (4, 4.2), arrowstyle='-|>', mutation_scale=20, 
                                        linewidth=3, color='#f39c12', alpha=0.8)
        ax.add_patch(highlight_arrow)
    
    # Phase 5: Agent learning
    else:
        # Circular motion around the learning cycle
        angle = (frame - 60) / 10 * np.pi * 2
        radius = 0.5
        learning_pos_x = 5.5 + radius * np.cos(angle)
        learning_pos_y = 2 + radius * np.sin(angle)
        
        # Draw learning path
        if (frame - 60) % 10 == 0:  # Create fresh path every few frames
            learning_path = Path(
                [(5.5 + radius * np.cos((frame-60)/10*np.pi*2 + i*np.pi/8), 
                  2 + radius * np.sin((frame-60)/10*np.pi*2 + i*np.pi/8)) 
                 for i in range(-8, 1)],
                [Path.MOVETO] + [Path.LINETO] * 8
            )
            path_patch = patches.PathPatch(learning_path, facecolor='none', 
                                         edgecolor='#8e44ad', alpha=0.8, linewidth=2)
            ax.add_patch(path_patch)
        
        learning_dot = Circle((learning_pos_x, learning_pos_y), 0.15, fill=True, color='#8e44ad')
        ax.add_patch(learning_dot)
        
        # Show learning indicator
        ax.text(5.5, 2, "Learning", ha='center', va='center', fontsize=10, 
              color='white', fontweight='bold',
              bbox=dict(facecolor='#8e44ad', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Update learning circle position for continuous cycle visualization
    circle_angle = frame / 10 * np.pi
    circle_x = 5.5 + 0.5 * np.cos(circle_angle)
    circle_y = 2 + 0.5 * np.sin(circle_angle)
    learning_circle.set_center((circle_x, circle_y))
    
    return ax.patches

# Create animation
ani = animation.FuncAnimation(fig, update, frames=75, interval=100, blit=False)

# Save as GIF
ani.save(os.path.join(output_dir, 'rl_cycle.gif'), writer='pillow', fps=10, dpi=100)

print(f"GIF saved to {os.path.join(output_dir, 'rl_cycle.gif')}") 