#!/usr/bin/env python3
"""
This script creates a simplified visual taxonomy of reinforcement learning algorithms.
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
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Define colors with reduced opacity for a cleaner look
value_color = "#3498db"     # Blue
policy_color = "#e74c3c"    # Red
actor_critic_color = "#9b59b6"  # Purple

# Helper function to create a box with a title and content
def create_box(x, y, width, height, title, content, color, alpha=0.15, fontsize=12):
    # Create the box with a subtle border
    rect = patches.Rectangle((x, y), width, height, facecolor=color, alpha=alpha, 
                            edgecolor=color, linewidth=1, zorder=1)
    ax.add_patch(rect)
    
    # Add title
    ax.text(x + width/2, y + height - 0.3, title, ha='center', va='center', 
           fontsize=fontsize, fontweight='bold', color=color, zorder=2)
    
    # Add content
    if isinstance(content, list):
        for i, item in enumerate(content):
            ax.text(x + 0.3, y + height - 0.7 - i*0.4, "• " + item,
                  fontsize=fontsize-2, va='center', zorder=2)
    else:
        ax.text(x + width/2, y + height/2, content, ha='center', va='center', 
               fontsize=fontsize-2, zorder=2)
    
    return rect

# Helper function to create a connector arrow
def create_arrow(start, end, color, style='->', connectionstyle='arc3,rad=0.0', linewidth=1.2, zorder=0):
    arrow = patches.FancyArrowPatch(
        start, end, arrowstyle=style, color=color, alpha=0.6,
        connectionstyle=connectionstyle, linewidth=linewidth, zorder=zorder
    )
    ax.add_patch(arrow)
    return arrow

# Add title at the top with more space below it
ax.text(5, 5.5, "Taxonomy of Reinforcement Learning Algorithms", ha='center', fontsize=16, fontweight='bold')

# Adjusted y-positions to move boxes down and create more space from the title
top_boxes_y = 3.5  # Moved down from 4.0
actor_critic_y = 1.7  # Moved down from 2.0

# Create the main paradigm boxes with better sizing for content
value_box = create_box(1.5, top_boxes_y, 3, 1.5, "Value-Based Methods", [
    "Estimate value functions",
    "Derive policy from values",
    "Examples: Q-Learning, DQN"
], value_color, fontsize=13)

policy_box = create_box(5.5, top_boxes_y, 3, 1.5, "Policy-Based Methods", [
    "Directly optimize policy",
    "No value function required", 
    "Examples: REINFORCE, TRPO"
], policy_color, fontsize=13)

actor_critic_box = create_box(3.5, actor_critic_y, 3, 1.5, "Actor-Critic Methods", [
    "Combine value and policy",
    "Reduce variance",
    "Examples: A2C, PPO, SAC"
], actor_critic_color, fontsize=13)

# Adjust arrow positions to connect the repositioned boxes
# Value to Actor-Critic
create_arrow((3, top_boxes_y), (4.2, top_boxes_y - 0.8), value_color, connectionstyle='arc3,rad=-0.2')

# Policy to Actor-Critic
create_arrow((7, top_boxes_y), (5.8, top_boxes_y - 0.8), policy_color, connectionstyle='arc3,rad=0.2')

# Save the figure with improved layout
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rl_algorithms_taxonomy.png'), dpi=300, bbox_inches='tight')

print(f"RL algorithms taxonomy saved to {os.path.join(output_dir, 'rl_algorithms_taxonomy.png')}") 