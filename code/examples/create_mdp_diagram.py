#!/usr/bin/env python3
"""
This script creates a Markov Decision Process (MDP) diagram
illustrating states, actions, transitions, and rewards.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create figure and axis
plt.figure(figsize=(12, 8))  # Increased figure size for better scaling
plt.axis('off')

# Add nodes (states)
states = ["S0", "S1", "S2", "S3", "S4"]
positions = {
    "S0": (2, 4),    # Adjusted positions to create more space
    "S1": (5, 6.5),
    "S2": (8, 4),
    "S3": (5, 1.5),
    "S4": (11, 4)
}

# Add edges (actions and transitions)
edges = [
    ("S0", "S1", {"action": "a0", "reward": "+1", "prob": "0.7"}),
    ("S0", "S3", {"action": "a0", "reward": "0", "prob": "0.3"}),
    ("S0", "S2", {"action": "a1", "reward": "+2", "prob": "0.8"}),
    ("S0", "S3", {"action": "a1", "reward": "-1", "prob": "0.2"}),
    ("S1", "S2", {"action": "a0", "reward": "+1", "prob": "0.6"}),
    ("S1", "S0", {"action": "a0", "reward": "0", "prob": "0.4"}),
    ("S1", "S4", {"action": "a1", "reward": "+10", "prob": "0.3"}),
    ("S1", "S2", {"action": "a1", "reward": "-2", "prob": "0.7"}),
    ("S2", "S4", {"action": "a0", "reward": "+5", "prob": "0.4"}), 
    ("S2", "S1", {"action": "a0", "reward": "0", "prob": "0.6"}),
    ("S2", "S0", {"action": "a1", "reward": "-1", "prob": "0.1"}),
    ("S2", "S3", {"action": "a1", "reward": "+3", "prob": "0.9"}),
    ("S3", "S0", {"action": "a0", "reward": "0", "prob": "1.0"}),
    ("S3", "S2", {"action": "a1", "reward": "+2", "prob": "0.5"}),
    ("S3", "S1", {"action": "a1", "reward": "+1", "prob": "0.5"}),
    ("S4", "S4", {"action": "a0", "reward": "+1", "prob": "1.0"}),
    ("S4", "S2", {"action": "a1", "reward": "0", "prob": "0.8"}),
    ("S4", "S3", {"action": "a1", "reward": "-5", "prob": "0.2"})
]

# Draw the graph
plt.title("Markov Decision Process (MDP)", fontsize=20)  # Larger title

# Draw state nodes
for state, pos in positions.items():
    circle = plt.Circle(pos, 0.7, color='skyblue', ec='blue', alpha=0.8, zorder=10)  # Larger circles
    plt.gca().add_patch(circle)
    plt.text(pos[0], pos[1], state, ha='center', va='center', fontsize=14, fontweight='bold', zorder=20)  # Larger text
    
    # Add terminal state double circle for S4
    if state == "S4":
        term_circle = plt.Circle(pos, 0.9, color='none', ec='blue', lw=2, zorder=5)  # Larger terminal circle
        plt.gca().add_patch(term_circle)
        plt.text(pos[0], pos[1]-1.2, "Terminal", ha='center', va='center', fontsize=12, color='darkblue')

# Draw edges (transitions)
curved_edges = [
    ("S0", "S3", 0.3),
    ("S1", "S0", 0.3),
    ("S2", "S1", 0.3),
    ("S3", "S1", 0.3),
    ("S4", "S3", 0.3)
]

curved_edges_dict = {(u, v): c for u, v, c in curved_edges}

for u, v, data in edges:
    start_pos = positions[u]
    end_pos = positions[v]
    
    # Calculate position for the midpoint (for labels)
    if u == v:  # Self-loop
        mid_x = start_pos[0] + 1.2
        mid_y = start_pos[1] + 1.2
        
        # Draw self-loop
        loop = patches.FancyArrowPatch(
            start_pos, (start_pos[0]+1.2, start_pos[1]+0.8),
            connectionstyle=f"arc3,rad=0.6", arrowstyle='->', 
            color='green', lw=2, alpha=0.7, zorder=0
        )
        plt.gca().add_patch(loop)
    else:
        # Determine if this is a curved edge
        rad = curved_edges_dict.get((u, v), 0)
        
        # Draw the arrow
        arrow = patches.FancyArrowPatch(
            start_pos, end_pos,
            connectionstyle=f"arc3,rad={rad}", arrowstyle='->', 
            color='green', lw=2, alpha=0.7, zorder=0
        )
        plt.gca().add_patch(arrow)
        
        # Calculate midpoint for label
        if rad != 0:
            # For curved edges, adjust the midpoint location
            mid_x = (start_pos[0] + end_pos[0]) / 2 + (1.5 * rad * (end_pos[1] - start_pos[1]))
            mid_y = (start_pos[1] + end_pos[1]) / 2 - (1.5 * rad * (end_pos[0] - start_pos[0]))
        else:
            # For straight edges
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
    
    # Add label with action, reward, and probability
    label = f"{data['action']}\nR={data['reward']}\np={data['prob']}"
    plt.text(mid_x, mid_y, label, ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3', ec='darkgreen'),
             fontsize=10, color='darkgreen', zorder=15)  # Improved label visibility

# Set plot limits with some padding
padding = 2.0  # Increased padding
all_x = [pos[0] for pos in positions.values()]
all_y = [pos[1] for pos in positions.values()]
plt.xlim(min(all_x) - padding, max(all_x) + padding)
plt.ylim(min(all_y) - padding, max(all_y) + padding)

# Add a title and description at the bottom
plt.figtext(0.5, 0.01, "Each transition shows: action (a0/a1), reward (R), and probability (p)", 
           ha='center', fontsize=12, style='italic')

# Save the diagram
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mdp_diagram.png'), dpi=300, bbox_inches='tight')

print(f"MDP diagram saved to {os.path.join(output_dir, 'mdp_diagram.png')}") 