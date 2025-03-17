#!/usr/bin/env python3

"""
This script creates two minimalistic animated GIFs for the multi-agent RL section:
1. Prisoner's Dilemma Game Theory: Shows agents adopting different strategies and payoffs
2. Predator-Prey Coordination: Shows multiple agents learning to coordinate to achieve a goal
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Arrow, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import os

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_prisoners_dilemma_gif(filename='prisoners_dilemma.gif', fps=5):
    """
    Creates an animated visualization of the Prisoner's Dilemma game,
    showing how different strategies evolve over time.
    """
    # Create figure with improved dimensions for better spacing
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
    
    # Setup for the animation - track game state
    rounds = ["Initial", "Round 1", "Round 2", "Round 3", "Round 4", "Round 5"]
    strategies = [
        ('C', 'C'),  # Start with cooperation
        ('C', 'D'),  # Player 2 defects
        ('D', 'D'),  # Player 1 also defects
        ('D', 'C'),  # Player 2 tries cooperation
        ('C', 'C'),  # Back to cooperation
        ('C', 'C')   # Stable cooperation
    ]
    
    scores = {
        1: [0],  # Player 1 score history
        2: [0]   # Player 2 score history
    }
    
    highlighted_cells = [
        None,   # Initial
        ('C', 'D'),  # Round 1
        ('D', 'D'),  # Round 2
        ('D', 'C'),  # Round 3
        ('C', 'C'),  # Round 4
        ('C', 'C')   # Round 5
    ]
    
    # Function to draw the matrix
    def draw_matrix(ax, highlight_cell=None, frame=0):
        ax.clear()
        ax.axis('off')
        
        # Draw title with round information
        title = rounds[frame] if frame < len(rounds) else rounds[-1]
        ax.text(5, 7.5, f"Prisoner's Dilemma: {title}", fontsize=14, fontweight='bold', ha='center')
        
        # Draw the scores for each player
        ax.text(2, 7, f"Score: {scores[1][-1]}", fontsize=12, ha='center')
        ax.text(8, 7, f"Score: {scores[2][-1]}", fontsize=12, ha='center')
        
        # Matrix borders - create a larger matrix with more space
        cell_size = 2.0
        
        # Draw the matrix border
        matrix_left = 3
        matrix_bottom = 2.5
        matrix_width = 4
        matrix_height = 4
        
        # Draw matrix grid lines
        rect = Rectangle((matrix_left, matrix_bottom), matrix_width, matrix_height,
                         facecolor='none', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        
        # Horizontal and vertical lines
        ax.plot([matrix_left, matrix_left + matrix_width], [matrix_bottom + matrix_height/2, matrix_bottom + matrix_height/2], 
                color='black', linewidth=1.5)
        ax.plot([matrix_left + matrix_width/2, matrix_left + matrix_width/2], [matrix_bottom, matrix_bottom + matrix_height], 
                color='black', linewidth=1.5)
        
        # Cell labels with adjusted positions for clarity
        ax.text(matrix_left - 0.5, matrix_bottom + matrix_height/4, "Defect", fontsize=10, va='center', ha='right', rotation=90)
        ax.text(matrix_left - 0.5, matrix_bottom + 3*matrix_height/4, "Cooperate", fontsize=10, va='center', ha='right', rotation=90)
        
        ax.text(matrix_left + matrix_width/4, matrix_bottom + matrix_height + 0.3, "Cooperate", fontsize=10, ha='center')
        ax.text(matrix_left + 3*matrix_width/4, matrix_bottom + matrix_height + 0.3, "Defect", fontsize=10, ha='center')
        
        # Add agent labels
        ax.text(matrix_left - 1, matrix_bottom + matrix_height/2, "Agent 1", fontsize=11, rotation=90, ha='center', va='center')
        ax.text(matrix_left + matrix_width/2, matrix_bottom - 0.7, "Agent 2", fontsize=11, ha='center')
        
        # Draw the payoffs in each cell with better spacing
        payoff_positions = {
            ('C', 'C'): (matrix_left + matrix_width/4, matrix_bottom + 3*matrix_height/4),  # Top-left
            ('C', 'D'): (matrix_left + 3*matrix_width/4, matrix_bottom + 3*matrix_height/4),  # Top-right
            ('D', 'C'): (matrix_left + matrix_width/4, matrix_bottom + matrix_height/4),  # Bottom-left
            ('D', 'D'): (matrix_left + 3*matrix_width/4, matrix_bottom + matrix_height/4)   # Bottom-right
        }
        
        for cell, (x, y) in payoff_positions.items():
            payoff = payoffs[cell]
            # Draw payoffs
            payoff_text = f"({payoff[0]}, {payoff[1]})"
            ax.text(x, y, payoff_text, ha='center', va='center', fontsize=10)
            
            # Highlight the current cell
            if cell == highlight_cell:
                circle = Circle((x, y), 0.5, fill=True, color='yellow', alpha=0.3, zorder=0)
                ax.add_patch(circle)
        
        # Draw agent choice indicators
        if frame > 0 and frame < len(strategies) + 1:
            strategy = strategies[frame-1]
            # Draw indicator for Agent 1's choice
            agent1_choice = 'C' if strategy[0] == 'C' else 'D'
            agent1_y = matrix_bottom + 3*matrix_height/4 if strategy[0] == 'C' else matrix_bottom + matrix_height/4
            agent1_circle = Circle((matrix_left - 1.5, agent1_y), 0.4, fill=True, color='#77dd77' if agent1_choice == 'C' else '#ff6961')
            ax.add_patch(agent1_circle)
            ax.text(matrix_left - 1.5, agent1_y, "1", ha='center', va='center', fontsize=10, color='white', fontweight='bold')
            
            # Draw indicator for Agent 2's choice
            agent2_choice = 'C' if strategy[1] == 'C' else 'D'
            agent2_x = matrix_left + matrix_width/4 if strategy[1] == 'C' else matrix_left + 3*matrix_width/4
            agent2_circle = Circle((agent2_x, matrix_bottom - 1.5), 0.4, fill=True, color='#77dd77' if agent2_choice == 'C' else '#ff6961')
            ax.add_patch(agent2_circle)
            ax.text(agent2_x, matrix_bottom - 1.5, "2", ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
    # Animation update function
    def update(frame):
        # Update scores based on strategies
        if frame > 0 and frame < len(strategies) + 1:
            p1_add, p2_add = payoffs[strategies[frame-1]]
            scores[1].append(scores[1][-1] + p1_add)
            scores[2].append(scores[2][-1] + p2_add)
        
        # Draw everything
        highlight = highlighted_cells[frame] if frame < len(highlighted_cells) else None
        draw_matrix(ax, highlight, frame)
        
        return ax.get_children()
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(rounds), interval=1000, blit=True
    )
    
    # Save as GIF with higher DPI for better quality
    anim.save(os.path.join(output_dir, filename), writer='pillow', fps=fps, dpi=120)
    print(f"Prisoner's Dilemma GIF saved to {os.path.join(output_dir, filename)}")

def create_predator_prey_gif(filename='predator_prey_coordination.gif', fps=8):
    """
    Creates an animated visualization of predator-prey coordination,
    showing how multiple predator agents learn to surround and capture a prey.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Grid parameters
    grid_size = 6
    cell_size = 1.0
    grid_left = 2
    grid_bottom = 1.5
    
    # Agent parameters
    prey_color = '#8CC152'  # Green
    predator_color = '#E9573F'  # Red
    
    # Stages of learning
    stages = [
        "Initial Random Movements",
        "Learning Positioning",
        "Developing Coordination",
        "Surrounding Strategy"
    ]
    
    # Define predator positions for each frame
    # Each frame represents a step in the learning process
    num_frames_per_stage = 5
    
    # Initial random movements (scattered)
    predator_positions_stage1 = [
        [(3, 2), (7, 3), (5, 6)],  # Frame 1
        [(3, 3), (6, 4), (4, 6)],  # Frame 2
        [(2, 3), (6, 3), (4, 5)],  # Frame 3
        [(2, 2), (5, 3), (3, 5)],  # Frame 4
        [(3, 2), (4, 3), (3, 4)]   # Frame 5
    ]
    
    # Learning positioning (getting closer to prey)
    predator_positions_stage2 = [
        [(3, 3), (5, 2), (6, 4)],  # Frame 6
        [(3, 3), (5, 3), (5, 5)],  # Frame 7
        [(3, 4), (4, 3), (6, 4)],  # Frame 8
        [(4, 4), (5, 2), (6, 3)],  # Frame 9
        [(4, 3), (6, 3), (5, 5)]   # Frame 10
    ]
    
    # Developing coordination (more structured movements)
    predator_positions_stage3 = [
        [(3, 3), (6, 3), (4, 5)],  # Frame 11
        [(3, 4), (5, 3), (4, 5)],  # Frame 12
        [(4, 4), (5, 4), (3, 5)],  # Frame 13
        [(4, 3), (6, 4), (4, 5)],  # Frame 14
        [(3, 3), (5, 3), (4, 4)]   # Frame 15
    ]
    
    # Surrounding strategy (encircling the prey)
    predator_positions_stage4 = [
        [(3, 3), (5, 3), (4, 5)],  # Frame 16
        [(3, 4), (5, 4), (4, 3)],  # Frame 17
        [(4, 5), (5, 3), (3, 3)],  # Frame 18
        [(3, 3), (5, 3), (4, 5)],  # Frame 19
        [(3, 4), (5, 4), (4, 3)]   # Frame 20
    ]
    
    all_predator_positions = (
        predator_positions_stage1 + 
        predator_positions_stage2 + 
        predator_positions_stage3 + 
        predator_positions_stage4
    )
    
    # Prey positions (gradually becomes surrounded)
    prey_positions = [
        (4, 4),  # Initial position
    ]
    
    # Generate prey positions with slight random movement
    np.random.seed(42)  # For reproducibility
    for i in range(1, len(all_predator_positions)):
        prev_x, prev_y = prey_positions[-1]
        
        # As learning progresses, prey movement becomes more constrained
        if i < num_frames_per_stage:
            # More random movement in early stages
            new_x = max(0, min(grid_size-1, prev_x + np.random.choice([-1, 0, 1])))
            new_y = max(0, min(grid_size-1, prev_y + np.random.choice([-1, 0, 1])))
        elif i < 2*num_frames_per_stage:
            # Less movement as predators learn
            new_x = max(0, min(grid_size-1, prev_x + np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])))
            new_y = max(0, min(grid_size-1, prev_y + np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])))
        else:
            # Very constrained movement as predators surround
            new_x = max(0, min(grid_size-1, prev_x + np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])))
            new_y = max(0, min(grid_size-1, prev_y + np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])))
        
        # Check if the new position is occupied by predators
        while (new_x, new_y) in all_predator_positions[i]:
            new_x = prev_x
            new_y = prev_y
        
        prey_positions.append((new_x, new_y))
    
    def update(frame):
        # Frame number determines the step in movement sequence
        ax.clear()
        ax.axis('off')
        
        # Determine current stage
        stage_idx = min(frame // num_frames_per_stage, len(stages) - 1)
        stage_text = stages[stage_idx]
        
        # Draw title with learning stage
        ax.text(5, 7.5, "Predator-Prey Coordination", fontsize=14, fontweight='bold', ha='center')
        ax.text(5, 6.8, stage_text, fontsize=12, ha='center', style='italic')
        
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
        
        # Draw the prey
        prey_x, prey_y = prey_positions[min(frame, len(prey_positions)-1)]
        prey_circle = Circle(
            (grid_left + (prey_x + 0.5) * cell_size, 
             grid_bottom + (prey_y + 0.5) * cell_size),
            0.35 * cell_size, fill=True, color=prey_color
        )
        ax.add_patch(prey_circle)
        ax.text(grid_left + (prey_x + 0.5) * cell_size, 
               grid_bottom + (prey_y + 0.5) * cell_size,
               "P", color='white', ha='center', va='center', fontweight='bold')
        
        # Draw the predators
        predator_pos = all_predator_positions[min(frame, len(all_predator_positions)-1)]
        for i, (px, py) in enumerate(predator_pos):
            predator_circle = Circle(
                (grid_left + (px + 0.5) * cell_size, 
                 grid_bottom + (py + 0.5) * cell_size),
                0.35 * cell_size, fill=True, color=predator_color
            )
            ax.add_patch(predator_circle)
            ax.text(grid_left + (px + 0.5) * cell_size, 
                   grid_bottom + (py + 0.5) * cell_size,
                   str(i+1), color='white', ha='center', va='center', fontweight='bold')
        
        # Add legend
        ax.add_patch(Circle((8.5, 2.0), 0.25, fill=True, color=predator_color))
        ax.text(9.0, 2.0, "Predator", va='center', fontsize=10)
        
        ax.add_patch(Circle((8.5, 1.5), 0.25, fill=True, color=prey_color))
        ax.text(9.0, 1.5, "Prey", va='center', fontsize=10)
        
        # Add explanation based on the stage
        if stage_idx == 0:
            explanation = "Predators move randomly\nwith no coordination"
        elif stage_idx == 1:
            explanation = "Predators learn to\napproach the prey"
        elif stage_idx == 2:
            explanation = "Predators develop basic\ncoordination patterns"
        else:
            explanation = "Predators learn to\nsurround and trap prey"
            
        ax.text(8.5, 3.0, explanation, va='center', fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        return ax.get_children()
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(all_predator_positions), interval=200, blit=True
    )
    
    # Save as GIF with higher DPI for better quality
    anim.save(os.path.join(output_dir, filename), writer='pillow', fps=fps, dpi=120)
    print(f"Predator-Prey Coordination GIF saved to {os.path.join(output_dir, filename)}")

if __name__ == "__main__":
    print("Creating Prisoner's Dilemma Game Theory GIF...")
    create_prisoners_dilemma_gif()
    
    print("Creating Predator-Prey Coordination GIF...")
    create_predator_prey_gif()
    
    print("GIF creation complete!") 