#!/usr/bin/env python3

"""
This script creates visualizations for the Procedural Generation and Generalization section:
1. A diagram showing different procedural generation techniques
2. A grid visualization of Procgen environments
3. A comparison of generalization techniques in reinforcement learning
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_procedural_generation_techniques_diagram():
    """
    Creates a diagram showing different procedural generation techniques and their applications.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Add title
    ax.text(5, 6.7, "Procedural Generation Techniques in RL", ha='center', 
            fontsize=18, fontweight='bold')
    
    # Create three main boxes for the primary techniques
    techniques = [
        {"name": "Random Seed Mechanisms", "x": 1.5, "y": 4.5, "width": 2.2, "height": 1.2, 
         "color": "#3498db", "elements": ["Seed-based generation", "Parameter distributions", "Constrained randomness"]},
        {"name": "Domain Randomization", "x": 5, "y": 4.5, "width": 2.2, "height": 1.2, 
         "color": "#e74c3c", "elements": ["Physics properties", "Visual properties", "Task parameters"]},
        {"name": "Curriculum Generation", "x": 8.5, "y": 4.5, "width": 2.2, "height": 1.2, 
         "color": "#2ecc71", "elements": ["Performance-based", "Novelty-based", "Learning progress"]}
    ]
    
    # Draw the technique boxes
    for tech in techniques:
        # Main box
        rect = patches.Rectangle((tech["x"] - tech["width"]/2, tech["y"] - tech["height"]/2), 
                                tech["width"], tech["height"], 
                                facecolor=tech["color"], alpha=0.2, 
                                edgecolor=tech["color"], linewidth=2)
        ax.add_patch(rect)
        
        # Title
        ax.text(tech["x"], tech["y"] + tech["height"]/2 - 0.2, tech["name"], 
                ha='center', va='center', fontsize=13, fontweight='bold', color=tech["color"])
        
        # Elements
        for i, elem in enumerate(tech["elements"]):
            y_pos = tech["y"] + 0.1 - (i+1) * 0.3
            ax.text(tech["x"], y_pos, f"• {elem}", ha='center', va='center', fontsize=10)
    
    # Define application areas
    applications = [
        {"name": "Level Generation", "x": 1.5, "y": 2.5, "r": 0.8, "color": "#3498db", 
         "desc": "Creating varied game levels\nand play spaces", "connects_to": [0]},
        {"name": "Sim-to-Real Transfer", "x": 5, "y": 2.5, "r": 0.8, "color": "#e74c3c", 
         "desc": "Bridging simulation and\nreality gaps", "connects_to": [1]},
        {"name": "Adaptive Learning", "x": 8.5, "y": 2.5, "r": 0.8, "color": "#2ecc71", 
         "desc": "Optimizing learning\nprogression", "connects_to": [2]},
        {"name": "Robust Generalization", "x": 5, "y": 0.8, "r": 1.0, "color": "#9b59b6", 
         "desc": "Training agents that adapt\nto novel scenarios", "connects_to": [0, 1, 2, 3, 4, 5]}
    ]
    
    # Draw the application areas
    for i, app in enumerate(applications):
        # Main circle
        circle = plt.Circle((app["x"], app["y"]), app["r"], 
                           facecolor=app["color"], alpha=0.2, 
                           edgecolor=app["color"], linewidth=2)
        ax.add_patch(circle)
        
        # Title
        ax.text(app["x"], app["y"] + 0.3, app["name"], 
                ha='center', va='center', fontsize=12, fontweight='bold', color=app["color"])
        
        # Description
        ax.text(app["x"], app["y"] - 0.1, app["desc"], 
                ha='center', va='center', fontsize=9)
    
    # Connect techniques to applications
    for i, tech in enumerate(techniques):
        tech_pos = (tech["x"], tech["y"] - tech["height"]/2)
        
        # Find applications that connect to this technique
        for j, app in enumerate(applications):
            if i in app["connects_to"]:
                app_pos = (app["x"], app["y"] + app["r"])
                
                # Draw arrow
                arrow = patches.FancyArrowPatch(tech_pos, app_pos, 
                                              arrowstyle='->', color=tech["color"],
                                              connectionstyle='arc3,rad=0.05', 
                                              linewidth=1.5, alpha=0.7)
                ax.add_patch(arrow)
    
    # Connect all application areas to the central "robust generalization" node
    central_app = applications[-1]  # The Robust Generalization node
    for i, app in enumerate(applications[:-1]):  # Exclude the last (central) app
        start_pos = (app["x"], app["y"] - app["r"])
        end_pos = (central_app["x"], central_app["y"] + central_app["r"])
        
        # Adjust for left/right positioning
        if app["x"] < central_app["x"]:
            start_pos = (app["x"] + app["r"] * 0.7, app["y"] - app["r"] * 0.7)
            end_pos = (central_app["x"] - central_app["r"] * 0.7, central_app["y"] + central_app["r"] * 0.7)
        elif app["x"] > central_app["x"]:
            start_pos = (app["x"] - app["r"] * 0.7, app["y"] - app["r"] * 0.7)
            end_pos = (central_app["x"] + central_app["r"] * 0.7, central_app["y"] + central_app["r"] * 0.7)
        
        # Draw arrow
        arrow = patches.FancyArrowPatch(start_pos, end_pos, 
                                      arrowstyle='->', color=app["color"],
                                      connectionstyle='arc3,rad=0.1', 
                                      linewidth=1.5, alpha=0.7)
        ax.add_patch(arrow)
    
    # Add explanation text
    explanation = (
        "Procedural generation in RL involves systematic variation of environment elements to create diverse training scenarios.\n"
        "These techniques work together to develop robust, adaptable agents that can generalize to novel situations."
    )
    ax.text(5, 6.2, explanation, ha='center', va='center', fontsize=10, 
           style='italic', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'procedural_generation_techniques.png'), dpi=300, bbox_inches='tight')
    print(f"Procedural generation techniques diagram saved to {os.path.join(output_dir, 'procedural_generation_techniques.png')}")
    plt.close()

def create_procgen_environments_grid():
    """
    Creates a grid visualization of the 16 Procgen environments.
    """
    # Define the grid
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(4, 4, hspace=0.4, wspace=0.4)
    
    # Title
    fig.suptitle("Procgen Benchmark Suite: Procedurally Generated Environments", 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Environment names and brief descriptions
    environments = [
        {"name": "CoinRun", "desc": "Platform game testing\ngeneralization", 
         "color": "#3498db", "icon": "⚪"},
        {"name": "StarPilot", "desc": "Space shooter with\nvarying enemies", 
         "color": "#e74c3c", "icon": "★"},
        {"name": "CaveFlyer", "desc": "Navigation in\nprocedural caves", 
         "color": "#2ecc71", "icon": "➤"},
        {"name": "Dodgeball", "desc": "Competitive game with\nmoving obstacles", 
         "color": "#f39c12", "icon": "◯"},
        {"name": "FruitBot", "desc": "Collection game with\nmoving targets", 
         "color": "#9b59b6", "icon": "✿"},
        {"name": "Chaser", "desc": "Pursuit game with\ncomplex dynamics", 
         "color": "#1abc9c", "icon": "♦"},
        {"name": "Miner", "desc": "Resource gathering in\ndynamic environments", 
         "color": "#d35400", "icon": "⬥"},
        {"name": "Jumper", "desc": "Platformer with\nvarying physics", 
         "color": "#2980b9", "icon": "▲"},
        {"name": "Leaper", "desc": "Precision jumping\nover obstacles", 
         "color": "#c0392b", "icon": "⟨⟩"},
        {"name": "Maze", "desc": "Navigation in\nrandom mazes", 
         "color": "#27ae60", "icon": "◱"},
        {"name": "BigFish", "desc": "Size-based\npredator-prey", 
         "color": "#8e44ad", "icon": "◗"},
        {"name": "Heist", "desc": "Key collection with\nstrategic planning", 
         "color": "#16a085", "icon": "⚿"},
        {"name": "Climber", "desc": "Vertical platformer\nwith variable gravity", 
         "color": "#e67e22", "icon": "⇧"},
        {"name": "Plunder", "desc": "Resource competition\nwith opponents", 
         "color": "#3498db", "icon": "⚔"},
        {"name": "Ninja", "desc": "Precision control\nwith moving hazards", 
         "color": "#e74c3c", "icon": "✪"},
        {"name": "BossFight", "desc": "Pattern recognition\nand adaptation", 
         "color": "#2ecc71", "icon": "⍟"}
    ]
    
    # Create a grid of environment visualizations
    for i, env in enumerate(environments):
        row, col = i // 4, i % 4
        ax = fig.add_subplot(gs[row, col])
        
        # Create a "screenshot" representation
        # Background
        ax.set_facecolor("#f8f9fa")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Game elements representation (simplified)
        # Background gradient
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        gradient = np.repeat(gradient, 100, axis=0)
        ax.imshow(gradient, cmap=plt.cm.get_cmap('Spectral'), 
                 extent=[0, 10, 0, 10], alpha=0.3, aspect='auto')
        
        # Game-specific elements
        # Character/player
        player = plt.Circle((3, 3), 0.7, color=env["color"], alpha=0.8)
        ax.add_patch(player)
        
        # Add some game elements based on the game type
        if i % 4 == 0:  # Platform-style games
            # Ground
            ax.add_patch(patches.Rectangle((0, 0), 10, 1.5, color='gray', alpha=0.4))
            # Platforms
            for p in range(2):
                x = np.random.uniform(1, 7)
                width = np.random.uniform(1, 3)
                y = np.random.uniform(3, 7)
                ax.add_patch(patches.Rectangle((x, y), width, 0.5, color='gray', alpha=0.4))
                
        elif i % 4 == 1:  # Shooter-style games
            # Enemies
            for e in range(3):
                x = np.random.uniform(6, 9)
                y = np.random.uniform(2, 8)
                size = np.random.uniform(0.3, 0.6)
                ax.add_patch(plt.Circle((x, y), size, color='red', alpha=0.4))
                
        elif i % 4 == 2:  # Collection-style games
            # Items to collect
            for c in range(4):
                x = np.random.uniform(1, 9)
                y = np.random.uniform(1, 9)
                ax.text(x, y, "✦", fontsize=12, ha='center', va='center', color='gold')
                
        else:  # Maze/navigation style
            # Walls/obstacles
            for w in range(3):
                x = np.random.uniform(1, 7)
                y = np.random.uniform(1, 7)
                width = np.random.uniform(0.5, 2)
                height = np.random.uniform(0.5, 2)
                ax.add_patch(patches.Rectangle((x, y), width, height, color='black', alpha=0.4))
        
        # Add environment icon
        ax.text(3, 3, env["icon"], fontsize=14, ha='center', va='center', color='white', 
               fontweight='bold')
        
        # Border
        ax.add_patch(patches.Rectangle((0, 0), 10, 10, fill=False, 
                                    edgecolor=env["color"], linewidth=2))
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add environment name and description
        ax.set_title(env["name"], fontsize=11, fontweight='bold', pad=5)
        ax.text(5, -1.2, env["desc"], ha='center', va='center', fontsize=8)
    
    # Add explanation text
    fig.text(0.5, 0.02, 
            "Procgen environments feature procedural generation of levels, obstacles, and visual elements\n"
            "to test agents' ability to generalize beyond their training experience.",
            ha='center', va='center', fontsize=10, style='italic')
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'procgen_environments.png'), dpi=300, bbox_inches='tight')
    print(f"Procgen environments grid saved to {os.path.join(output_dir, 'procgen_environments.png')}")
    plt.close()

def create_generalization_techniques_comparison():
    """
    Create a simplified visual comparison of four generalization techniques:
    1. Data augmentation (showing original and transformed observations)
    2. Regularization (showing neural network with dropout)
    3. Uncertainty-based exploration (showing agent exploring unknown areas)
    4. Meta-learning (showing quick adaptation to new environments)
    """
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Set title for the entire figure
    plt.suptitle('Generalization Techniques in Reinforcement Learning', fontsize=20, y=0.98)
    
    # 1. Data Augmentation - purely visual
    ax1 = fig.add_subplot(gs[0, 0])
    create_data_augmentation_visual(ax1)
    ax1.set_title('Data Augmentation', fontsize=16)
    ax1.axis('off')
    
    # 2. Regularization - neural network with dropout
    ax2 = fig.add_subplot(gs[0, 1])
    create_regularization_visual(ax2)
    ax2.set_title('Regularization', fontsize=16)
    ax2.axis('off')
    
    # 3. Uncertainty Exploration - agent exploring unknown areas
    ax3 = fig.add_subplot(gs[1, 0])
    create_uncertainty_exploration_visual(ax3)
    ax3.set_title('Uncertainty-Based Exploration', fontsize=16)
    ax3.axis('off')
    
    # 4. Meta-Learning - adaptation across environments
    ax4 = fig.add_subplot(gs[1, 1])
    create_meta_learning_visual(ax4)
    ax4.set_title('Meta-Learning', fontsize=16)
    ax4.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'generalization_techniques_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved generalization techniques comparison to {os.path.join(output_dir, 'generalization_techniques_comparison.png')}")

def create_data_augmentation_visual(ax):
    """Create a visual showing image transformations with minimal text"""
    # Original image (simple grid world)
    grid_size = 5
    
    # Create original image (simple grid world)
    original = np.ones((grid_size, grid_size, 3))
    # Add a wall (red)
    original[1:4, 2] = [0.8, 0.2, 0.2]
    # Add goal (green)
    original[1, 4] = [0.2, 0.8, 0.2]
    # Add agent (blue)
    original[3, 0] = [0.2, 0.2, 0.8]
    
    # Create transformed versions
    # 1. Rotated
    rotated = np.rot90(original.copy())
    # 2. Flipped
    flipped = np.fliplr(original.copy())
    # 3. Color jittered (slightly change colors)
    jittered = original.copy() * np.random.uniform(0.7, 1.3, size=original.shape)
    jittered = np.clip(jittered, 0, 1)
    # 4. Noisy (add noise)
    noisy = original.copy() + np.random.normal(0, 0.1, size=original.shape)
    noisy = np.clip(noisy, 0, 1)
    
    # Place images in grid
    # Original in center
    ax.imshow(original, extent=[-0.5, 0.5, -0.5, 0.5], interpolation='nearest')
    
    # Transformed versions around it
    ax.imshow(rotated, extent=[0.7, 1.7, -0.5, 0.5], interpolation='nearest')
    ax.imshow(flipped, extent=[-0.5, 0.5, 0.7, 1.7], interpolation='nearest')
    ax.imshow(jittered, extent=[-1.7, -0.7, -0.5, 0.5], interpolation='nearest')
    ax.imshow(noisy, extent=[-0.5, 0.5, -1.7, -0.7], interpolation='nearest')
    
    # Connect original to transformed versions with arrows
    ax.arrow(0, 0, 0.6, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(0, 0, 0, 0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(0, 0, -0.6, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(0, 0, 0, -0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

def create_regularization_visual(ax):
    """Create a visual showing neural network with dropout - simplified version"""
    # Neural network layout
    n_inputs = 6
    n_hidden = 8
    n_outputs = 3
    
    # Node positions
    input_pos = np.array([[0, i] for i in range(n_inputs)])
    hidden_pos = np.array([[1, i*0.9] for i in range(n_hidden)])
    output_pos = np.array([[2, i+1.5] for i in range(n_outputs)])
    
    # Randomly drop some hidden nodes (to show dropout)
    dropout_mask = np.random.random(n_hidden) > 0.3
    
    # Draw edges first (connections between nodes)
    for i, ip in enumerate(input_pos):
        for j, hp in enumerate(hidden_pos):
            if dropout_mask[j]:  # Only connect to active nodes
                ax.plot([ip[0], hp[0]], [ip[1], hp[1]], 'gray', alpha=0.3)
    
    for i, hp in enumerate(hidden_pos):
        if dropout_mask[i]:  # Only connect from active nodes
            for j, op in enumerate(output_pos):
                ax.plot([hp[0], op[0]], [hp[1], op[1]], 'gray', alpha=0.3)
    
    # Draw nodes
    # Input layer
    ax.scatter(input_pos[:, 0], input_pos[:, 1], s=100, color='#3498db', zorder=10)
    
    # Hidden layer - active nodes are purple, dropped nodes are red X marks
    for i, hp in enumerate(hidden_pos):
        if dropout_mask[i]:
            ax.scatter(hp[0], hp[1], s=100, color='#9b59b6', zorder=10)
        else:
            # Draw X mark for dropped nodes
            ax.scatter(hp[0], hp[1], s=100, color='white', edgecolors='#e74c3c', linewidth=2, zorder=10)
            ax.plot([hp[0]-0.15, hp[0]+0.15], [hp[1]-0.15, hp[1]+0.15], color='#e74c3c', linewidth=2, zorder=11)
            ax.plot([hp[0]-0.15, hp[0]+0.15], [hp[1]+0.15, hp[1]-0.15], color='#e74c3c', linewidth=2, zorder=11)
    
    # Output layer
    ax.scatter(output_pos[:, 0], output_pos[:, 1], s=100, color='#2ecc71', zorder=10)
    
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, n_inputs)

def create_uncertainty_exploration_visual(ax):
    """Create a maze with uncertainty visualization"""
    # Create a simple maze grid (1 = wall, 0 = passage)
    maze_size = 10
    maze = np.ones((maze_size, maze_size))
    
    # Create paths in the maze
    # Start at center
    maze[4:7, 4:7] = 0
    # Create some corridors
    maze[5, :7] = 0  # horizontal corridor
    maze[2:9, 3] = 0  # vertical corridor
    maze[2, 3:8] = 0  # upper horizontal
    maze[8, 3:8] = 0  # lower horizontal
    maze[2:9, 7] = 0  # right vertical
    
    # Add some dead ends and loops
    maze[3, 1:3] = 0
    maze[7, 1:3] = 0
    maze[1, 7] = 0
    maze[9, 7] = 0
    
    # Uncertainty heatmap (higher = more uncertain)
    uncertainty = np.ones((maze_size, maze_size))
    
    # Areas closer to center (explored) have lower uncertainty
    for i in range(maze_size):
        for j in range(maze_size):
            dist = np.sqrt((i - 5)**2 + (j - 5)**2)
            if maze[i, j] == 0:  # only for valid paths
                uncertainty[i, j] = min(1.0, dist / 5)
            else:
                uncertainty[i, j] = np.nan  # walls are masked
    
    # Create a custom colormap: blue (low uncertainty) to red (high uncertainty)
    cmap = LinearSegmentedColormap.from_list('uncertainty', 
                                           [(0, '#3498db'),  # blue
                                            (0.5, '#f39c12'),  # orange
                                            (1, '#e74c3c')])  # red
    
    # Plot the maze with uncertainty
    ax.imshow(uncertainty, cmap=cmap, alpha=0.8)
    
    # Add agent at center
    ax.scatter(5, 5, s=150, color='#2ecc71', marker='o', edgecolors='black', zorder=10)
    
    # Draw exploration paths (agent's movement)
    # Main path
    path = np.array([
        [5, 5], [5, 4], [5, 3], [5, 2], [5, 1],  # left
        [5, 5], [4, 5], [3, 5], [2, 5], [1, 5],  # up
        [5, 5], [6, 5], [7, 5], [8, 5], [9, 5],  # down
        [5, 5], [5, 6], [5, 7], [5, 8], [5, 9],  # right
    ])
    
    # Plot paths with decreasing opacity to show sequence
    segments = [path[:5], path[5:10], path[10:15], path[15:20]]
    alphas = [0.9, 0.7, 0.5, 0.3]
    
    for segment, alpha in zip(segments, alphas):
        ax.plot(segment[:, 1], segment[:, 0], 'k-', alpha=alpha, linewidth=2)
    
    # No grid and axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

def create_meta_learning_visual(ax):
    """Create a visual showing meta-learning across environments - simplified with no text"""
    # Define environment positions - centered around meta-learner
    env_positions = {
        'A': (-1.5, 1.5),
        'B': (0, 2),
        'C': (1.5, 1.5),
        'New': (0, -2),
        'Meta': (0, 0)
    }
    
    # Define colors for each environment
    env_colors = {
        'A': '#3498db',  # blue
        'B': '#e74c3c',  # red
        'C': '#2ecc71',  # green
        'New': '#f39c12',  # orange
        'Meta': '#9b59b6'  # purple
    }
    
    # Draw environments as circles
    for env, pos in env_positions.items():
        if env != 'Meta':
            # Regular environments are smaller circles
            circle = plt.Circle(pos, 0.7, color=env_colors[env], alpha=0.5, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
        else:
            # The meta-learner is a larger circle
            circle = plt.Circle(pos, 1.0, color=env_colors[env], alpha=0.6, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
    
    # Draw arrows connecting meta-learner to environments
    for env, pos in env_positions.items():
        if env != 'Meta':
            # Calculate arrow start and end
            meta_pos = env_positions['Meta']
            
            if env == 'New':
                # New environment gets a double arrow (bidirectional) - showing fast adaptation
                arrow = patches.FancyArrowPatch(
                    meta_pos, pos, 
                    connectionstyle="arc3,rad=0.1",
                    arrowstyle='->', 
                    color=env_colors[env], 
                    linewidth=2,
                    alpha=0.8
                )
                ax.add_patch(arrow)
                
                # Return arrow - showing what meta-learner learns from new environment
                arrow = patches.FancyArrowPatch(
                    pos, meta_pos, 
                    connectionstyle="arc3,rad=0.1",
                    arrowstyle='->', 
                    color=env_colors[env], 
                    linewidth=2, 
                    alpha=0.8
                )
                ax.add_patch(arrow)
            else:
                # Training environments feed into meta-learner
                arrow = patches.FancyArrowPatch(
                    pos, meta_pos, 
                    connectionstyle="arc3,rad=0.1",
                    arrowstyle='->', 
                    color=env_colors[env], 
                    linewidth=2,
                    alpha=0.8
                )
                ax.add_patch(arrow)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

if __name__ == "__main__":
    print("Generating procedural generation techniques diagram...")
    create_procedural_generation_techniques_diagram()
    
    print("Generating Procgen environments grid...")
    create_procgen_environments_grid()
    
    print("Generating generalization techniques comparison...")
    create_generalization_techniques_comparison()
    
    print("All visualizations for Section 8 have been created successfully!") 