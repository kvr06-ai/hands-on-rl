#!/usr/bin/env python3

"""
This script creates visualizations for the Multi-Agent Environments section:
1. Markov Games formulation (competitive/cooperative matrices)
2. Predator-Prey scenarios
3. Multi-agent communication visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
from matplotlib.colors import LinearSegmentedColormap

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Common style settings for all visualizations
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})

def create_multiagent_framework_diagram():
    """
    Create a simplified visualization showing only the Multi-Agent RL Framework
    (without the game theory matrices)
    """
    # Create figure with significantly reduced size (7x4 inches)
    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Add title
    ax.text(5, 5.5, "Multi-Agent RL Framework", 
            ha='center', fontsize=10, fontweight='bold')
    
    # Simplified agent representation
    agent_positions = [(2, 3), (5, 3), (8, 3)]
    agent_colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (pos, color) in enumerate(zip(agent_positions, agent_colors)):
        # Draw agent circle
        agent = plt.Circle(pos, 0.6, color=color, alpha=0.7)
        ax.add_patch(agent)
        ax.text(pos[0], pos[1], f"Agent {i+1}", ha='center', va='center', 
                color='white', fontsize=8, fontweight='bold')
    
    # Draw environment
    env_rect = patches.Rectangle((3, 0.8), 4, 1, linewidth=1, edgecolor='black', 
                               facecolor='lightgray', alpha=0.6)
    ax.add_patch(env_rect)
    ax.text(5, 1.3, "Environment", ha='center', fontsize=8, fontweight='bold')
    
    # Draw arrows for agent-environment interaction
    # Action arrows
    for i, pos in enumerate(agent_positions):
        ax.annotate("", xy=(5, 1.5), xytext=(pos[0], pos[1]-0.6),
                   arrowprops=dict(arrowstyle="->", lw=1.2, color=agent_colors[i], alpha=0.7))
    
    # State/reward arrow - just one simplified arrow for clarity
    ax.annotate("", xy=(5, 2.5), xytext=(5, 1.8),
               arrowprops=dict(arrowstyle="->", lw=1.2, color='gray', alpha=0.7))
    
    # Add labels for arrows
    ax.text(3.5, 2.0, "Actions", ha='center', fontsize=7, rotation=40)
    ax.text(5.5, 2.3, "States, Rewards", ha='center', fontsize=7)
    
    # Add a brief explanation of what multi-agent RL is
    explanation = (
        "Multiple agents interacting with a shared environment and with each other,\n"
        "each pursuing individual or collective goals."
    )
    plt.figtext(0.5, 0.08, explanation, ha='center', fontsize=7,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiagent_framework.png'), dpi=150, bbox_inches='tight')
    print(f"Multi-agent framework diagram saved to {os.path.join(output_dir, 'multiagent_framework.png')}")
    plt.close()

def create_game_theory_matrices():
    """
    Create a separate visualization showing just the game theory matrices 
    (competitive and cooperative)
    """
    # Create figure with appropriate size for matrices
    fig, ax = plt.subplots(figsize=(7, 6))  # Increased height for more spacing
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)  # Increased height
    ax.axis('off')
    
    # Add title for game matrix
    ax.text(5, 8.5, "Game Theory Matrices in Multi-Agent RL", ha='center', fontsize=12, fontweight='bold')
    
    # Draw competitive game matrix (top)
    ax.text(5, 7.5, "Competitive Game (Zero-Sum)", ha='center', fontsize=10, fontweight='bold')
    comp_matrix = np.array([[(-1,1), (-5,5)], [(0,0), (-2,2)]])
    
    # More space between matrices
    draw_game_matrix(ax, 5, 5.5, comp_matrix, cell_size=1.8, 
                    highlight_cell=(1,0), highlight_color='blue', 
                    highlight_label="Nash\nEquilibrium")
    
    # Add explanation for competitive game - positioned better
    competitive_explanation = (
        "Zero-sum games where one agent's gain is another's loss.\n"
        "Agents have opposing objectives."
    )
    ax.text(5, 3.8, competitive_explanation, ha='center', fontsize=8,
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Draw cooperative game matrix (bottom) - more spacing
    ax.text(5, 3.2, "Cooperative Game (Team Reward)", ha='center', fontsize=10, fontweight='bold')
    coop_matrix = np.array([[(3,3), (0,0)], [(0,0), (5,5)]])
    draw_game_matrix(ax, 5, 1.8, coop_matrix, cell_size=1.8, 
                    highlight_cell=(1,1), highlight_color='green', 
                    highlight_label="Pareto\nOptimal")
    
    # Add explanation for cooperative game - positioned better
    cooperative_explanation = (
        "Cooperative games where agents benefit from working together.\n"
        "Agents share common objectives."
    )
    ax.text(5, 0.3, cooperative_explanation, ha='center', fontsize=8,
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Add row/column labels to matrices for clarity
    # For competitive game - spaced better to avoid overlap
    comp_left = 5 - 1.8
    comp_top = 5.5 + 1.8
    
    # Row labels (left side)
    ax.text(comp_left - 0.8, comp_top - 0.9, "Agent 1\nAction A", 
            ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(comp_left - 0.8, comp_top - 2.7, "Agent 1\nAction B", 
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Column labels (top)
    ax.text(comp_left + 0.9, comp_top + 0.8, "Agent 2\nAction A", 
            ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(comp_left + 2.7, comp_top + 0.8, "Agent 2\nAction B", 
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    # For cooperative game - spaced better to avoid overlap
    coop_left = 5 - 1.8
    coop_top = 1.8 + 1.8
    
    # Row labels (left side)
    ax.text(coop_left - 0.8, coop_top - 0.9, "Agent 1\nAction A", 
            ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(coop_left - 0.8, coop_top - 2.7, "Agent 1\nAction B", 
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Column labels (top)
    ax.text(coop_left + 0.9, coop_top + 0.8, "Agent 2\nAction A", 
            ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(coop_left + 2.7, coop_top + 0.8, "Agent 2\nAction B", 
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout(pad=2.0)  # Added padding for better spacing
    plt.savefig(os.path.join(output_dir, 'multiagent_game_matrices.png'), dpi=150, bbox_inches='tight')
    print(f"Game theory matrices saved to {os.path.join(output_dir, 'multiagent_game_matrices.png')}")
    plt.close()

def draw_game_matrix(ax, center_x, center_y, payoffs, cell_size=1.5, highlight_cell=None, highlight_color=None, highlight_label=None):
    """Helper function to draw a 2x2 game matrix centered at (center_x, center_y)"""
    # Calculate top-left corner
    left = center_x - cell_size
    top = center_y + cell_size
    
    # Draw grid with thicker lines
    # Horizontal lines
    ax.plot([left, left + 2*cell_size], [top, top], 'k-', lw=1.5)
    ax.plot([left, left + 2*cell_size], [top - cell_size, top - cell_size], 'k-', lw=1.5) 
    ax.plot([left, left + 2*cell_size], [top - 2*cell_size, top - 2*cell_size], 'k-', lw=1.5)
    
    # Vertical lines
    ax.plot([left, left], [top, top - 2*cell_size], 'k-', lw=1.5)
    ax.plot([left + cell_size, left + cell_size], [top, top - 2*cell_size], 'k-', lw=1.5)
    ax.plot([left + 2*cell_size, left + 2*cell_size], [top, top - 2*cell_size], 'k-', lw=1.5)
    
    # Highlight cell if specified with more visible highlighting
    if highlight_cell is not None:
        i, j = highlight_cell
        rect = patches.Rectangle((left + j*cell_size, top - (i+1)*cell_size),
                              cell_size, cell_size, linewidth=2, edgecolor=highlight_color,
                              facecolor=highlight_color, alpha=0.2)  # Increased alpha and linewidth
        ax.add_patch(rect)
        
        if highlight_label:
            # Position the label to avoid overlapping with cell content
            ax.text(left + (j+0.5)*cell_size, top - (i+0.2)*cell_size - 0.7,
                   highlight_label, ha='center', fontsize=9, color=highlight_color, 
                   fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, pad=0.2, boxstyle='round'))
    
    # Add payoffs with better formatting
    for i in range(2):
        for j in range(2):
            cell_center_x = left + (j+0.5)*cell_size
            cell_center_y = top - (i+0.5)*cell_size
            payoff = payoffs[i, j]
            if isinstance(payoff, tuple):
                # Format the tuple payoffs more clearly
                ax.text(cell_center_x, cell_center_y, f"{payoff[0]}, {payoff[1]}", 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            else:
                ax.text(cell_center_x, cell_center_y, str(payoff), 
                       ha='center', va='center', fontsize=10, fontweight='bold')

def create_predator_prey_visualization():
    """
    Create a simplified visualization of predator-prey dynamics in multi-agent systems
    """
    # Create figure with reduced size
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    
    # Set background color and remove axes
    ax.set_facecolor('#f8f9fa')
    ax.axis('off')
    
    # Create boundary
    boundary = patches.Rectangle((1, 1), 8, 5.5, linewidth=1, edgecolor='black',
                               facecolor='none', alpha=0.7)
    ax.add_patch(boundary)
    
    # Define simplified obstacles
    obstacles = [
        patches.Rectangle((2, 2), 1.5, 2, facecolor='gray', alpha=0.5),
        patches.Rectangle((7, 3), 1.5, 2, facecolor='gray', alpha=0.5),
    ]
    
    for obstacle in obstacles:
        ax.add_patch(obstacle)
    
    # Define predator and prey positions
    predator_positions = [(2.5, 5), (7, 2), (8, 5)]
    prey_position = (5, 3.5)
    
    # Draw predators
    for i, pos in enumerate(predator_positions):
        predator = plt.Circle(pos, 0.3, facecolor='#e74c3c', edgecolor='black', alpha=0.7)
        ax.add_patch(predator)
        ax.text(pos[0], pos[1], f"P{i+1}", ha='center', va='center', color='white', fontsize=7, fontweight='bold')
    
    # Draw prey
    prey = plt.Circle(prey_position, 0.3, facecolor='#2ecc71', edgecolor='black', alpha=0.7)
    ax.add_patch(prey)
    ax.text(prey_position[0], prey_position[1], "Prey", ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Draw predator strategy - triangle showing encirclement
    strategy_points = [(3.5, 4), (7, 4), (5, 2.5)]
    strategy_polygon = patches.Polygon(strategy_points, closed=True, 
                                     facecolor='#3498db', alpha=0.15,
                                     edgecolor='#3498db', linestyle='-')
    ax.add_patch(strategy_polygon)
    
    # Add "Encirclement Strategy" label
    strategy_center = np.mean(strategy_points, axis=0)
    ax.text(strategy_center[0], strategy_center[1], "Encirclement", 
           ha='center', va='center', fontsize=7)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='#e74c3c', edgecolor='black', alpha=0.7, label='Predators'),
        patches.Patch(facecolor='#2ecc71', edgecolor='black', alpha=0.7, label='Prey'),
        patches.Patch(facecolor='#3498db', alpha=0.15, label='Coordination')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95),
             ncol=3, frameon=False, fontsize=7)
    
    # Add title
    ax.set_title('Multi-Agent Predator-Prey Environment', fontsize=10, pad=10)
    
    # Add simpler description at bottom with only key points
    description = (
        "• Predators coordinate to encircle prey\n"
        "• Limited visibility due to obstacles\n"
        "• Mix of cooperation and competition"
    )
    plt.figtext(0.5, 0.05, description, ha='center', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predator_prey_environment.png'), dpi=150, bbox_inches='tight')
    print(f"Predator-prey visualization saved to {os.path.join(output_dir, 'predator_prey_environment.png')}")
    plt.close()

def create_algorithm_comparison_chart():
    """
    Create a simplified comparison chart showing different MARL algorithms
    """
    # Algorithm data
    algorithms = [
        'Independent Q-Learning', 
        'MADDPG', 
        'QMIX', 
        'MAPPO'
    ]
    
    # Simplified metrics with fewer rows
    metrics = {
        'Handles Continuous Actions': [0, 1, 0, 1],
        'Centralized Training': [0, 1, 1, 1],
        'Partial Observability': [0.3, 0.8, 1, 0.8],
        'Stability': [0.4, 0.7, 0.8, 0.9]
    }
    
    # Create figure with reduced size
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    # Create the heatmap
    data = np.array(list(metrics.values()))
    
    # Create a cleaner, simpler heatmap
    im = ax.imshow(data, cmap='Blues', aspect='auto')
    
    # Adjust colors for binary values and add text annotations
    binary_metrics = ['Handles Continuous Actions', 'Centralized Training']
    binary_indices = [list(metrics.keys()).index(m) for m in binary_metrics]
    
    for i in range(len(metrics)):
        for j in range(len(algorithms)):
            if i in binary_indices:
                text = "Yes" if data[i, j] > 0.5 else "No"
                color = 'white' if data[i, j] > 0.7 else 'black'
                ax.text(j, i, text, ha="center", va="center", color=color,
                       fontsize=7, fontweight='bold')
            else:
                # Simplified rating display
                rating = int(round(data[i, j] * 10))
                color = 'white' if data[i, j] > 0.7 else 'black'
                ax.text(j, i, f"{rating}/10", ha="center", va="center", color=color, fontsize=7)
    
    # Set labels
    ax.set_xticks(np.arange(len(algorithms)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(algorithms, fontsize=8)
    ax.set_yticklabels(metrics.keys(), fontsize=8)
    
    # Rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    
    # Add a title
    ax.set_title("Multi-Agent RL Algorithm Comparison", fontsize=10, pad=10)
    
    # Add a simplified key characteristics box
    key_features = {
        'Independent Q-Learning': 'Simple but non-stationary',
        'MADDPG': 'Centralized critic with decentralized actors',
        'QMIX': 'Monotonic value function factorization',
        'MAPPO': 'Policy gradient with centralized value function'
    }
    
    text_y = 0.05
    plt.figtext(0.5, text_y, 
               " | ".join([f"{algo}: {feat}" for algo, feat in key_features.items()]),
               fontsize=6, ha='center',
               bbox=dict(facecolor='whitesmoke', alpha=0.7, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'marl_algorithm_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"MARL algorithm comparison saved to {os.path.join(output_dir, 'marl_algorithm_comparison.png')}")
    plt.close()

def create_communication_protocol_visualization():
    """
    Create a simplified visualization showing communication protocols in multi-agent systems
    """
    # Create figure with reduced size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    
    # Communication network visualization in left subplot
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5)
    ax1.axis('off')
    ax1.set_title("Emergent Communication Network", fontsize=10)
    
    # Draw agents (simplified with fewer agents)
    agent_positions = {
        'Agent 1': (2, 2.5),
        'Agent 2': (5, 3.5),
        'Agent 3': (8, 2.5),
    }
    
    agent_colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Draw agents
    for i, (agent, pos) in enumerate(agent_positions.items()):
        circle = plt.Circle(pos, 0.5, facecolor=agent_colors[i], edgecolor='none', alpha=0.7)
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], agent.split()[-1], ha='center', va='center', 
                color='white', fontsize=7, fontweight='bold')
    
    # Draw communication links (simplified)
    communications = [
        ('Agent 1', 'Agent 2', 0.9),
        ('Agent 2', 'Agent 3', 0.7),
        ('Agent 3', 'Agent 1', 0.4),
    ]
    
    for source, target, strength in communications:
        source_pos = agent_positions[source]
        target_pos = agent_positions[target]
        
        # Draw arrow
        ax1.annotate("", xy=target_pos, xytext=source_pos,
                    arrowprops=dict(arrowstyle="->", color='gray', 
                                    alpha=0.7, linewidth=1 + strength,
                                    connectionstyle="arc3,rad=0.1"))
        
        # Add simple label
        mid_x = (source_pos[0] + target_pos[0]) / 2
        mid_y = (source_pos[1] + target_pos[1]) / 2
        
        # Add small offset to avoid overlapping with arrow
        offset_x = 0.2 * (target_pos[1] - source_pos[1])
        offset_y = -0.2 * (target_pos[0] - source_pos[0])
        
        ax1.text(mid_x + offset_x, mid_y + offset_y, f"{int(strength*10)}", 
                fontsize=7, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle='round'))
    
    # Communication protocol visualization in right subplot
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    ax2.axis('off')
    ax2.set_title("Learned Communication Protocol", fontsize=10)
    
    # Create a simplified message structure visualization
    message_components = [
        {'name': 'Header', 'bits': 3, 'color': '#3498db'},
        {'name': 'Agent ID', 'bits': 2, 'color': '#2ecc71'},
        {'name': 'Position', 'bits': 5, 'color': '#e74c3c'},
        {'name': 'Intent', 'bits': 3, 'color': '#f39c12'},
        {'name': 'CRC', 'bits': 2, 'color': '#7f8c8d'}
    ]
    
    # Draw the message structure
    total_bits = sum(comp['bits'] for comp in message_components)
    bit_width = 6.0 / total_bits  # Calculate width per bit to fill available space
    
    current_x = 2  # Starting position
    for component in message_components:
        width = component['bits'] * bit_width
        rect = patches.Rectangle((current_x, 2.5), width, 0.8, 
                               facecolor=component['color'], alpha=0.7)
        ax2.add_patch(rect)
        
        # Add component name
        ax2.text(current_x + width/2, 2.9, component['name'], 
                ha='center', va='center', fontsize=7, color='black')
        
        # Add bit count
        ax2.text(current_x + width/2, 2.7, f"{component['bits']} bits", 
                ha='center', va='center', fontsize=6, color='white')
        
        current_x += width
    
    # Add explanation
    explanation = (
        "Agents learn what information is valuable to share and how to encode it efficiently.\n"
        "Communication emerges through training rather than explicit design."
    )
    
    plt.figtext(0.5, 0.08, explanation, ha='center', fontsize=7,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'marl_communication_protocols.png'), dpi=150, bbox_inches='tight')
    print(f"MARL communication protocols visualization saved to {os.path.join(output_dir, 'marl_communication_protocols.png')}")
    plt.close()

if __name__ == "__main__":
    print("Generating multi-agent framework diagram...")
    create_multiagent_framework_diagram()
    
    print("Generating game theory matrices...")
    create_game_theory_matrices()
    
    print("Generating predator-prey visualization...")
    create_predator_prey_visualization()
    
    print("Generating algorithm comparison chart...")
    create_algorithm_comparison_chart()
    
    print("Generating communication protocol visualization...")
    create_communication_protocol_visualization()
    
    print("All multi-agent environment visualizations complete!") 