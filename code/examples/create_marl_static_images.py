#!/usr/bin/env python3

"""
This script creates static images that capture the most informative frames
from the multi-agent RL GIFs:
1. Prisoner's Dilemma Game Theory: Shows the key payoff structure and strategies
2. Predator-Prey Coordination: Shows coordinated predator behavior surrounding prey
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_prisoners_dilemma_static(filename='prisoners_dilemma_static.png'):
    """
    Creates a static visualization of the Prisoner's Dilemma game,
    showing the key strategies and payoffs in a clear format.
    """
    # Create figure with reduced dimensions to match article scaling
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define payoffs for the Prisoner's Dilemma
    payoffs = {
        ('C', 'C'): (3, 3),   # Both cooperate
        ('C', 'D'): (0, 5),   # Player 1 cooperates, Player 2 defects
        ('D', 'C'): (5, 0),   # Player 1 defects, Player 2 cooperates
        ('D', 'D'): (1, 1)    # Both defect
    }
    
    # For the static image, show a mixed state with some cooperation and defection
    # We'll highlight the Nash equilibrium and the optimal cooperative outcome
    
    # Draw title with smaller font
    ax.text(5, 7.5, "Prisoner's Dilemma Game Theory", fontsize=14, fontweight='bold', ha='center')
    
    # Add explanatory text with reduced size
    ax.text(5, 6.8, "A foundational model for cooperation vs. competition in multi-agent systems", 
            fontsize=10, ha='center', style='italic')
    
    # Matrix parameters - slightly smaller
    matrix_left = 3
    matrix_bottom = 2.5
    matrix_width = 3.8
    matrix_height = 3.8
    
    # Draw matrix grid lines
    rect = Rectangle((matrix_left, matrix_bottom), matrix_width, matrix_height,
                     facecolor='none', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Horizontal and vertical lines
    ax.plot([matrix_left, matrix_left + matrix_width], [matrix_bottom + matrix_height/2, matrix_bottom + matrix_height/2], 
            color='black', linewidth=1.5)
    ax.plot([matrix_left + matrix_width/2, matrix_left + matrix_width/2], [matrix_bottom, matrix_bottom + matrix_height], 
            color='black', linewidth=1.5)
    
    # Cell labels with adjusted positions for clarity - smaller fonts
    ax.text(matrix_left - 0.5, matrix_bottom + matrix_height/4, "Defect", fontsize=9, va='center', ha='right', rotation=90)
    ax.text(matrix_left - 0.5, matrix_bottom + 3*matrix_height/4, "Cooperate", fontsize=9, va='center', ha='right', rotation=90)
    
    ax.text(matrix_left + matrix_width/4, matrix_bottom + matrix_height + 0.3, "Cooperate", fontsize=9, ha='center')
    ax.text(matrix_left + 3*matrix_width/4, matrix_bottom + matrix_height + 0.3, "Defect", fontsize=9, ha='center')
    
    # Add agent labels - smaller
    ax.text(matrix_left - 1, matrix_bottom + matrix_height/2, "Agent 1", fontsize=10, rotation=90, ha='center', va='center')
    ax.text(matrix_left + matrix_width/2, matrix_bottom - 0.7, "Agent 2", fontsize=10, ha='center')
    
    # Draw the payoffs in each cell with better spacing
    payoff_positions = {
        ('C', 'C'): (matrix_left + matrix_width/4, matrix_bottom + 3*matrix_height/4),  # Top-left
        ('C', 'D'): (matrix_left + 3*matrix_width/4, matrix_bottom + 3*matrix_height/4),  # Top-right
        ('D', 'C'): (matrix_left + matrix_width/4, matrix_bottom + matrix_height/4),  # Bottom-left
        ('D', 'D'): (matrix_left + 3*matrix_width/4, matrix_bottom + matrix_height/4)   # Bottom-right
    }
    
    # Add cell descriptions and payoffs
    descriptions = {
        ('C', 'C'): "Mutual\nCooperation",
        ('C', 'D'): "Exploitation of\nCooperator",
        ('D', 'C'): "Exploitation of\nCooperator",
        ('D', 'D'): "Mutual\nDefection"
    }
    
    # Color scheme
    coop_color = '#77dd77'  # Green for cooperation
    defect_color = '#ff6961'  # Red for defection
    
    for cell, (x, y) in payoff_positions.items():
        payoff = payoffs[cell]
        
        # Background colors for cells
        if cell == ('C', 'C'):
            # Mutual cooperation - light green background
            cell_rect = Rectangle((matrix_left, matrix_bottom + matrix_height/2), matrix_width/2, matrix_height/2, 
                                facecolor=coop_color, alpha=0.2, edgecolor='none')
            ax.add_patch(cell_rect)
        elif cell == ('D', 'D'):
            # Mutual defection (Nash equilibrium) - highlighted
            cell_rect = Rectangle((matrix_left + matrix_width/2, matrix_bottom), matrix_width/2, matrix_height/2, 
                                facecolor=defect_color, alpha=0.2, edgecolor='none')
            ax.add_patch(cell_rect)
            # Add Nash equilibrium indicator
            ax.add_patch(Circle((x, y), 0.5, fill=True, color='#ffcc5c', alpha=0.3, zorder=0))
            ax.text(x, y - 0.7, "Nash Equilibrium", fontsize=8, ha='center', color='#996600')
        
        # Draw payoffs - smaller size
        payoff_text = f"({payoff[0]}, {payoff[1]})"
        ax.text(x, y + 0.25, payoff_text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add description - smaller
        ax.text(x, y - 0.25, descriptions[cell], ha='center', va='center', fontsize=8, style='italic')
    
    # Add cooperative optimum indicator - smaller
    ax.text(payoff_positions[('C', 'C')][0], payoff_positions[('C', 'C')][1] - 0.7, 
            "Cooperative Optimum", fontsize=8, ha='center', color='#336633')
    
    # Add legend explaining the dilemma - more compact
    legend_text = (
        "The Prisoner's Dilemma illustrates why rational agents\n"
        "might not cooperate even when it's in their collective interest.\n"
        "While mutual cooperation yields the best group outcome,\n"
        "each agent has individual incentive to defect."
    )
    ax.text(5, 1, legend_text, ha='center', fontsize=9,
           bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    # Save the image with reduced DPI to match article scaling
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    print(f"Prisoner's Dilemma static image saved to {os.path.join(output_dir, filename)}")
    plt.close()

def create_predator_prey_static(filename='predator_prey_static.png'):
    """
    Creates a static visualization of predator-prey coordination,
    showing how predator agents coordinate to surround a prey.
    """
    # Create figure with reduced dimensions
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Grid parameters - smaller
    grid_size = 6
    cell_size = 0.9
    grid_left = 2.5
    grid_bottom = 1.8
    
    # Agent parameters
    prey_color = '#8CC152'  # Green
    predator_color = '#E9573F'  # Red
    
    # Draw title with learning stage - smaller fonts
    ax.text(5, 7.5, "Predator-Prey Coordination in Multi-Agent RL", fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 6.8, "Emergent cooperative behavior between predator agents to surround prey", 
            fontsize=10, ha='center', style='italic')
    
    # Draw the grid
    for i in range(grid_size + 1):
        # Vertical lines
        ax.plot([grid_left + i * cell_size, grid_left + i * cell_size],
               [grid_bottom, grid_bottom + grid_size * cell_size],
               color='gray', linewidth=0.5, alpha=0.5)
        # Horizontal lines
        ax.plot([grid_left, grid_left + grid_size * cell_size],
               [grid_bottom + i * cell_size, grid_bottom + i * cell_size],
               color='gray', linewidth=0.5, alpha=0.5)
    
    # Predator positions - showing surrounding strategy
    predator_positions = [(3, 3), (5, 3), (4, 5)]
    
    # Prey position - surrounded
    prey_position = (4, 4)
    
    # Draw movements and coordination arrows
    arrows = [
        ((2.5, 3), (3, 3), "Initial\nPositioning"),
        ((5.5, 2.5), (5, 3), "Flanking\nMovement"),
        ((4, 5.5), (4, 5), "Blocking\nEscape")
    ]
    
    # Draw coordination arrows
    for i, ((start_x, start_y), (end_x, end_y), label) in enumerate(arrows):
        start_point = (grid_left + (start_x + 0.5) * cell_size, 
                      grid_bottom + (start_y + 0.5) * cell_size)
        end_point = (grid_left + (end_x + 0.5) * cell_size, 
                    grid_bottom + (end_y + 0.5) * cell_size)
        
        # Draw arrow
        arrow = FancyArrowPatch(
            start_point, end_point,
            arrowstyle='-|>', color='#3498db', linewidth=1.2,
            mutation_scale=12, zorder=0, alpha=0.7
        )
        ax.add_patch(arrow)
        
        # Add label offset from arrow - smaller font
        offset_x = -0.7 if i == 0 else (0.7 if i == 1 else 0)
        offset_y = 0 if i != 2 else -0.7
        ax.text(start_point[0] + offset_x, start_point[1] + offset_y, 
               label, ha='center', va='center', fontsize=8,
               bbox=dict(facecolor='white', alpha=0.7, edgecolor=None, boxstyle='round,pad=0.1'))
    
    # Add communication lines between predators
    for i, pos1 in enumerate(predator_positions):
        for j, pos2 in enumerate(predator_positions):
            if i < j:  # Only connect each pair once
                start_point = (grid_left + (pos1[0] + 0.5) * cell_size, 
                              grid_bottom + (pos1[1] + 0.5) * cell_size)
                end_point = (grid_left + (pos2[0] + 0.5) * cell_size, 
                            grid_bottom + (pos2[1] + 0.5) * cell_size)
                
                # Draw dashed line representing communication
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                       'k--', alpha=0.3, linewidth=0.8, zorder=0)
    
    # Draw the prey
    prey_x, prey_y = prey_position
    prey_circle = Circle(
        (grid_left + (prey_x + 0.5) * cell_size, 
         grid_bottom + (prey_y + 0.5) * cell_size),
        0.3 * cell_size, fill=True, color=prey_color
    )
    ax.add_patch(prey_circle)
    ax.text(grid_left + (prey_x + 0.5) * cell_size, 
           grid_bottom + (prey_y + 0.5) * cell_size,
           "P", color='white', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Draw escape paths (blocked)
    escape_paths = [
        ((4, 4), (3, 4), "Blocked"),
        ((4, 4), (5, 4), "Blocked"),
        ((4, 4), (4, 3), "Blocked")
    ]
    
    for (start_x, start_y), (end_x, end_y), label in escape_paths:
        start_point = (grid_left + (start_x + 0.5) * cell_size, 
                      grid_bottom + (start_y + 0.5) * cell_size)
        end_point = (grid_left + (end_x + 0.5) * cell_size, 
                    grid_bottom + (end_y + 0.5) * cell_size)
        
        # Draw blocked escape path
        arrow = FancyArrowPatch(
            start_point, end_point,
            arrowstyle='-|>', color='red', linewidth=0.8,
            mutation_scale=8, zorder=0, alpha=0.4
        )
        ax.add_patch(arrow)
        
        # Add small X to show blocked path
        mid_x = (start_point[0] + end_point[0]) / 2
        mid_y = (start_point[1] + end_point[1]) / 2
        ax.plot([mid_x-0.08, mid_x+0.08], [mid_y-0.08, mid_y+0.08], 'r-', linewidth=1.5, alpha=0.6)
        ax.plot([mid_x-0.08, mid_x+0.08], [mid_y+0.08, mid_y-0.08], 'r-', linewidth=1.5, alpha=0.6)
    
    # Draw the predators with ID numbers
    for i, (px, py) in enumerate(predator_positions):
        predator_circle = Circle(
            (grid_left + (px + 0.5) * cell_size, 
             grid_bottom + (py + 0.5) * cell_size),
            0.3 * cell_size, fill=True, color=predator_color
        )
        ax.add_patch(predator_circle)
        ax.text(grid_left + (px + 0.5) * cell_size, 
               grid_bottom + (py + 0.5) * cell_size,
               str(i+1), color='white', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Add legend with smaller fonts
    legend_x = 8.4
    ax.add_patch(Circle((legend_x, 3.0), 0.2, fill=True, color=predator_color))
    ax.text(legend_x + 0.4, 3.0, "Predator Agent", va='center', fontsize=8.5)
    
    ax.add_patch(Circle((legend_x, 2.5), 0.2, fill=True, color=prey_color))
    ax.text(legend_x + 0.4, 2.5, "Prey Agent", va='center', fontsize=8.5)
    
    # Add dashed line to legend
    ax.plot([legend_x - 0.2, legend_x + 0.2], [2.0, 2.0], 'k--', alpha=0.5)
    ax.text(legend_x + 0.4, 2.0, "Communication", va='center', fontsize=8.5)
    
    # Add explanation with smaller font
    explanation = (
        "In multi-agent reinforcement learning, predator agents can learn to:\n"
        "• Coordinate movements to surround prey\n"
        "• Block escape routes through strategic positioning\n"
        "• Develop emergent team behaviors without explicit programming\n"
        "• Communicate and share information for improved performance"
    )
    
    ax.text(5, 1.0, explanation, ha='center', va='center', fontsize=9,
           bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.4'))
    
    # Save the image with reduced DPI
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    print(f"Predator-Prey Coordination static image saved to {os.path.join(output_dir, filename)}")
    plt.close()

if __name__ == "__main__":
    print("Creating Prisoner's Dilemma Game Theory static image...")
    create_prisoners_dilemma_static()
    
    print("Creating Predator-Prey Coordination static image...")
    create_predator_prey_static()
    
    print("Static image creation complete!") 