#!/usr/bin/env python3

"""
This script creates visualizations for Section 10: Modern Simulation Platforms:
1. Isaac Gym parallel simulation visualization
2. Meta-World tasks visualization 
3. Traffic simulation visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_isaac_gym_visualization():
    """
    Creates a visualization showing Isaac Gym's parallel simulation capabilities.
    """
    print("Creating Isaac Gym parallel simulation visualization...")
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    plt.suptitle("Isaac Gym: GPU-Accelerated Physics Simulation", fontsize=20, y=0.98)
    
    # Create a 2x2 grid layout
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[0.7, 1.3], wspace=0.2, hspace=0.3)
    
    # Header with performance comparison
    ax_header = plt.subplot(gs[0, :])
    ax_header.set_title("Simulation Performance Comparison", fontsize=16, pad=10)
    ax_header.axis('off')
    
    # Create a bar chart comparing traditional vs parallel simulations
    ax_chart = plt.subplot(gs[1, 0])
    ax_chart.set_title("Training Throughput: Environments per Second", fontsize=14)
    
    # Data for the bars
    methods = ["Traditional\n(CPU, 1 env)", "IsaacGym\n(GPU, 4,096 envs)"]
    throughput = [100, 10000]  # Environments per second (approximate)
    
    # Create the bars with a log scale
    bars = ax_chart.bar(methods, throughput, color=['#1976D2', '#F57C00'], width=0.6)
    ax_chart.set_yscale('log')
    ax_chart.set_ylabel("Environments/second (log scale)")
    ax_chart.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax_chart.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f"{int(height):,}",
                    ha='center', va='bottom', fontsize=12)
    
    # Right subplot: parallel simulation visualization
    ax_parallel = plt.subplot(gs[1, 1])
    ax_parallel.set_title("4,096 Parallel Humanoid Environments", fontsize=14)
    ax_parallel.axis('equal')
    ax_parallel.set_xlim(0, 10)
    ax_parallel.set_ylim(0, 10)
    ax_parallel.axis('off')
    
    # Draw a grid of small robot figures to represent parallel environments
    grid_size = 16  # 16x16 grid of small environments
    grid_spacing = 9.0 / grid_size
    
    # Create a linear gradient colormap for training progress
    cmap = LinearSegmentedColormap.from_list('training_progress', 
                                          ['#B71C1C', '#F57F17', '#33691E'], N=256)
    
    # Draw small humanoid figures in a grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Skip some positions randomly to show variation
            if np.random.rand() > 0.9:
                continue
                
            # Position in the grid
            x = 0.5 + i * grid_spacing
            y = 0.5 + j * grid_spacing
            
            # Color based on "training progress"
            progress = np.random.rand()
            color = cmap(progress)
            
            # Draw a simplified humanoid
            # Head
            circle = plt.Circle((x, y + 0.25), 0.1, facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
            ax_parallel.add_patch(circle)
            
            # Body
            rect = plt.Rectangle((x - 0.1, y), 0.2, 0.2, facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
            ax_parallel.add_patch(rect)
            
            # Legs in random pose (to show variety)
            leg_angle1 = np.random.uniform(-30, 30)
            leg_angle2 = np.random.uniform(-30, 30)
            
            leg1 = patches.FancyArrowPatch((x - 0.05, y), (x - 0.15, y - 0.2), 
                                        arrowstyle='-', color='black', linewidth=1.5,
                                        connectionstyle=f'arc3,rad={np.radians(leg_angle1)/6}')
            leg2 = patches.FancyArrowPatch((x + 0.05, y), (x + 0.15, y - 0.2), 
                                        arrowstyle='-', color='black', linewidth=1.5,
                                        connectionstyle=f'arc3,rad={np.radians(leg_angle2)/6}')
            
            ax_parallel.add_patch(leg1)
            ax_parallel.add_patch(leg2)
            
            # Arms in random poses too
            arm_angle1 = np.random.uniform(-45, 45)
            arm_angle2 = np.random.uniform(-45, 45)
            
            arm1 = patches.FancyArrowPatch((x - 0.1, y + 0.15), (x - 0.2, y + 0.05), 
                                         arrowstyle='-', color='black', linewidth=1.5,
                                         connectionstyle=f'arc3,rad={np.radians(arm_angle1)/6}')
            arm2 = patches.FancyArrowPatch((x + 0.1, y + 0.15), (x + 0.2, y + 0.05), 
                                         arrowstyle='-', color='black', linewidth=1.5,
                                         connectionstyle=f'arc3,rad={np.radians(arm_angle2)/6}')
            
            ax_parallel.add_patch(arm1)
            ax_parallel.add_patch(arm2)
    
    # Add annotation about GPU acceleration
    ax_parallel.text(5, 0.2, "One forward pass computes all environments in parallel on GPU", 
                   ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add performance highlights in the header
    features = [
        "10,000+ environments per second",
        "100-1000x faster training",
        "Same physics fidelity as traditional simulators",
        "CUDA-accelerated collision detection & dynamics"
    ]
    
    for i, feature in enumerate(features):
        x_pos = 0.25 + (i // 2) * 0.5
        y_pos = 0.7 - (i % 2) * 0.3
        ax_header.text(x_pos, y_pos, f"• {feature}", fontsize=13, 
                     bbox=dict(facecolor='#E3F2FD', edgecolor='#90CAF9', boxstyle='round,pad=0.3'))
    
    # Save the figure
    output_path = os.path.join(output_dir, 'isaac_gym_parallel.png')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Isaac Gym visualization saved to {output_path}")

def create_metaworld_visualization():
    """
    Creates a visualization showing Meta-World's diverse manipulation tasks.
    """
    print("Creating Meta-World tasks visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    plt.suptitle("Meta-World: Benchmark for Meta-Reinforcement Learning", fontsize=20, y=0.98)
    
    # Create a grid layout - simplifying to just 9 tasks in a 3x3 grid
    gs = gridspec.GridSpec(4, 3, wspace=0.3, hspace=0.4)
    
    # Define a reduced set of representative Meta-World tasks
    tasks = [
        {"name": "Button-Press", "color": "#E57373", "difficulty": "Easy"},
        {"name": "Drawer-Open", "color": "#81C784", "difficulty": "Medium"},
        {"name": "Door-Open", "color": "#64B5F6", "difficulty": "Medium"},
        {"name": "Pick-Place", "color": "#FFB74D", "difficulty": "Hard"},
        {"name": "Push", "color": "#9575CD", "difficulty": "Easy"},
        {"name": "Hammer", "color": "#4DB6AC", "difficulty": "Hard"},
        {"name": "Peg-Insert", "color": "#F06292", "difficulty": "Hard"},
        {"name": "Window-Open", "color": "#DCE775", "difficulty": "Medium"},
        {"name": "Sweep", "color": "#A1887F", "difficulty": "Medium"}
    ]
    
    # Add chart in the first row showing task categorization
    ax_chart = plt.subplot(gs[0, :])
    ax_chart.set_title("ML-10 & ML-45 Task Sets by Difficulty", fontsize=16)
    
    # Group tasks by difficulty
    difficulties = ["Easy", "Medium", "Hard", "Very Hard"]
    difficulty_counts = {d: sum(1 for task in tasks if task["difficulty"] == d) for d in difficulties}
    
    # Create stacked bar chart for ML-10 (subset) and ML-45 (scaled up version)
    bar_width = 0.35
    x = np.arange(len(difficulties))
    
    # ML-10 bars (actual distribution from our task list)
    ml10_counts = [difficulty_counts[d] for d in difficulties]
    ax_chart.bar(x - bar_width/2, ml10_counts, bar_width, label='ML-10 (train)', color='#5C6BC0')
    
    # ML-45 bars (scaled up from our task list for illustration)
    ml45_counts = [count * 4 if count > 0 else 0 for count in ml10_counts]
    ax_chart.bar(x + bar_width/2, ml45_counts, bar_width, label='ML-45 (train+test)', color='#26A69A')
    
    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels(difficulties)
    ax_chart.set_ylabel("Number of Tasks")
    ax_chart.legend()
    ax_chart.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Draw task visualizations in a grid - now only 9 tasks (3x3)
    for i, task in enumerate(tasks):
        # Calculate the position in the 3x3 grid (starting from row 1)
        row = 1 + i // 3
        col = i % 3
        
        # Create subplot
        ax = plt.subplot(gs[row, col])
        ax.set_title(task["name"], fontsize=12)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Draw a simplified visualization of the task
        # Background (table/workspace)
        table = patches.Rectangle((1, 2), 8, 6, linewidth=1, edgecolor='#888888', 
                               facecolor='#EEEEEE', zorder=1)
        ax.add_patch(table)
        
        # Robot arm base
        base = patches.Rectangle((1, 1), 2, 1, linewidth=1, edgecolor='#555555', 
                              facecolor='#AAAAAA', zorder=2)
        ax.add_patch(base)
        
        # Draw different objects based on the task
        if task["name"] == "Button-Press":
            # Button
            button = patches.Circle((6, 5), 0.8, linewidth=1, edgecolor='#333333', 
                                 facecolor=task["color"], zorder=3)
            ax.add_patch(button)
            
            # Robot arm
            arm = patches.FancyArrowPatch((2, 2), (4.5, 4), arrowstyle='-', color='#555555', 
                                       linewidth=2, connectionstyle='arc3,rad=0.3', zorder=4)
            ax.add_patch(arm)
            
            # End effector
            endeff = patches.Circle((4.5, 4), 0.4, linewidth=1, edgecolor='#333333', 
                                 facecolor='#DDDDDD', zorder=5)
            ax.add_patch(endeff)
            
            # Arrow showing intended action
            action = patches.Arrow(4.5, 4, 1.5, 1, width=0.3, color='red', zorder=6)
            ax.add_patch(action)
            
        elif task["name"] == "Drawer-Open":
            # Drawer
            drawer = patches.Rectangle((5, 3), 3, 2, linewidth=1, edgecolor='#333333', 
                                    facecolor=task["color"], zorder=3)
            ax.add_patch(drawer)
            
            # Drawer handle
            handle = patches.Rectangle((5.5, 4), 0.5, 0.5, linewidth=1, edgecolor='#333333', 
                                    facecolor='#555555', zorder=4)
            ax.add_patch(handle)
            
            # Robot arm
            arm = patches.FancyArrowPatch((2, 2), (4.5, 4.2), arrowstyle='-', color='#555555', 
                                       linewidth=2, connectionstyle='arc3,rad=0.2', zorder=5)
            ax.add_patch(arm)
            
            # End effector
            endeff = patches.Circle((4.5, 4.2), 0.4, linewidth=1, edgecolor='#333333', 
                                 facecolor='#DDDDDD', zorder=6)
            ax.add_patch(endeff)
            
            # Arrow showing intended action
            action = patches.Arrow(6, 4.5, -1.5, 0, width=0.3, color='red', zorder=7)
            ax.add_patch(action)
            
        elif task["name"] == "Door-Open":
            # Door frame
            frame = patches.Rectangle((5, 3), 0.2, 3, linewidth=1, edgecolor='#333333', 
                                   facecolor='#777777', zorder=3)
            ax.add_patch(frame)
            
            # Door
            door = patches.Rectangle((5.2, 3), 2, 3, linewidth=1, edgecolor='#333333', 
                                  facecolor=task["color"], zorder=4)
            ax.add_patch(door)
            
            # Door handle
            handle = patches.Circle((6.7, 4.5), 0.3, linewidth=1, edgecolor='#333333', 
                                 facecolor='#555555', zorder=5)
            ax.add_patch(handle)
            
            # Robot arm
            arm = patches.FancyArrowPatch((2, 2), (5.5, 4.5), arrowstyle='-', color='#555555', 
                                       linewidth=2, connectionstyle='arc3,rad=0.2', zorder=6)
            ax.add_patch(arm)
            
            # End effector
            endeff = patches.Circle((5.5, 4.5), 0.4, linewidth=1, edgecolor='#333333', 
                                 facecolor='#DDDDDD', zorder=7)
            ax.add_patch(endeff)
            
            # Arrow showing intended action
            action = patches.Arrow(6.7, 4.5, 1, 0, width=0.3, color='red', zorder=8)
            ax.add_patch(action)
            
        else:
            # Generic object
            obj = patches.Rectangle((5, 4), 2, 1, linewidth=1, edgecolor='#333333', 
                                 facecolor=task["color"], zorder=3)
            ax.add_patch(obj)
            
            # Robot arm
            arm = patches.FancyArrowPatch((2, 2), (4, 4.5), arrowstyle='-', color='#555555', 
                                       linewidth=2, connectionstyle='arc3,rad=0.2', zorder=4)
            ax.add_patch(arm)
            
            # End effector
            endeff = patches.Circle((4, 4.5), 0.4, linewidth=1, edgecolor='#333333', 
                                 facecolor='#DDDDDD', zorder=5)
            ax.add_patch(endeff)
        
        # Add difficulty label
        difficulty_colors = {
            "Easy": "#4CAF50",
            "Medium": "#FFC107",
            "Hard": "#F44336",
            "Very Hard": "#9C27B0"
        }
        diff_color = difficulty_colors[task["difficulty"]]
        
        ax.text(5, 0.8, task["difficulty"], ha='center', fontsize=10, 
              bbox=dict(facecolor=diff_color, alpha=0.2, boxstyle='round,pad=0.2'))
    
    # Add subtitle about Meta-World
    fig.text(0.5, 0.02, "Meta-World provides standardized environments for meta-reinforcement learning research.",
           ha='center', va='bottom', fontsize=12, fontstyle='italic')
    
    # Save the figure
    output_path = os.path.join(output_dir, 'metaworld_tasks.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust for suptitle and subtitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Meta-World visualization saved to {output_path}")

def create_traffic_simulation_visualization():
    """
    Creates a visualization showing modern traffic simulation platforms.
    """
    print("Creating traffic simulation visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    plt.suptitle("Modern Traffic Simulation Platforms", fontsize=20, y=0.98)
    
    # Create a grid layout with different sections
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[0.35, 0.65], wspace=0.3, hspace=0.3)
    
    # Top section: Comparison of platforms
    ax_comparison = plt.subplot(gs[0, :])
    ax_comparison.set_title("SMARTS vs. Flow: Simulation Capabilities", fontsize=16)
    ax_comparison.axis('off')
    
    # Create a comparison table
    table_data = [
        ["Feature", "SMARTS", "Flow"],
        ["Focus", "Autonomous driving behaviors", "Traffic optimization & control"],
        ["Physics", "High-fidelity vehicles", "Macro traffic flow dynamics"],
        ["Scale", "100s of detailed agents", "1000s of simplified vehicles"],
        ["Integration", "SUMO & CARLA", "SUMO & Aimsun"]
    ]
    
    table = ax_comparison.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.2, 0.8, 0.8]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    
    # Color the header row
    for j, cell in enumerate(table_data[0]):
        table[(0, j)].set_facecolor('#E3F2FD')
        table[(0, j)].set_text_props(fontweight='bold')
    
    # Color platform columns
    for i in range(1, len(table_data)):
        table[(i, 1)].set_facecolor('#FFECB3')  # SMARTS
        table[(i, 2)].set_facecolor('#E8F5E9')  # Flow
    
    # Adjust cell sizes
    table.scale(1, 1.5)
    
    # Left bottom: SMARTS visualization
    ax_smarts = plt.subplot(gs[1, 0])
    ax_smarts.set_title("SMARTS: Multi-Agent Driving Scenarios", fontsize=14)
    ax_smarts.set_xlim(0, 10)
    ax_smarts.set_ylim(0, 10)
    ax_smarts.axis('off')
    
    # Draw a road network
    # Main highway
    highway = patches.Rectangle((0, 4), 10, 2, linewidth=1, edgecolor='black', 
                             facecolor='#EEEEEE', zorder=1)
    ax_smarts.add_patch(highway)
    
    # Lane markings
    for y in [5, 6]:
        for x in range(0, 10):
            if x % 2 == 0:
                line = patches.Rectangle((x, y), 0.8, 0.05, linewidth=0, facecolor='white', zorder=2)
                ax_smarts.add_patch(line)
    
    # Add vehicles with different behaviors
    # Ego vehicle (autonomous agent)
    ego_car = patches.Rectangle((2, 4.6), 0.8, 0.4, linewidth=1, edgecolor='black', 
                             facecolor='#4285F4', zorder=3)
    ax_smarts.add_patch(ego_car)
    
    # Draw a "perception zone" around ego vehicle
    perception = patches.Ellipse((2.4, 4.8), 3, 1.5, linewidth=1, edgecolor='#4285F4', 
                              facecolor='none', linestyle='--', zorder=2)
    ax_smarts.add_patch(perception)
    
    # Other vehicles
    car1 = patches.Rectangle((4, 4.6), 0.7, 0.4, linewidth=1, edgecolor='black', 
                          facecolor='#DB4437', zorder=3)
    car2 = patches.Rectangle((6, 5.6), 0.7, 0.4, linewidth=1, edgecolor='black', 
                          facecolor='#F4B400', zorder=3)
    car3 = patches.Rectangle((8, 4.6), 0.7, 0.4, linewidth=1, edgecolor='black', 
                          facecolor='#0F9D58', zorder=3)
    
    ax_smarts.add_patch(car1)
    ax_smarts.add_patch(car2)
    ax_smarts.add_patch(car3)
    
    # Add arrows showing trajectories and social behaviors
    # Ego vehicle trajectory
    ego_arrow = patches.FancyArrowPatch((3, 4.8), (4, 5.8), arrowstyle='->', color='#4285F4', 
                                      linewidth=1.5, connectionstyle='arc3,rad=0.3', zorder=4)
    ax_smarts.add_patch(ego_arrow)
    
    # Other vehicle reactions
    car1_arrow = patches.FancyArrowPatch((4.7, 4.8), (6, 4.8), arrowstyle='->', color='#DB4437', 
                                       linewidth=1.5, connectionstyle='arc3,rad=0', zorder=4)
    ax_smarts.add_patch(car1_arrow)
    
    # Annotations
    ax_smarts.text(2.4, 5.5, "Ego Agent", fontsize=10, ha='center', 
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    ax_smarts.text(5, 4.2, "Social Agents", fontsize=10, ha='center', 
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Key SMARTS features
    features_smarts = [
        "Multi-agent interaction",
        "Social behavior modeling",
        "Diverse scenarios"
    ]
    
    for i, feature in enumerate(features_smarts):
        ax_smarts.text(1, 8.5 - i*0.6, f"• {feature}", fontsize=10, ha='left')
    
    # Right bottom: Flow visualization
    ax_flow = plt.subplot(gs[1, 1])
    ax_flow.set_title("Flow: Traffic Control Optimization", fontsize=14)
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 10)
    ax_flow.axis('off')
    
    # Draw a more complex road network for Flow
    # Main horizontal highway
    highway_h = patches.Rectangle((0, 4), 10, 1.5, linewidth=1, edgecolor='black', 
                               facecolor='#EEEEEE', zorder=1)
    ax_flow.add_patch(highway_h)
    
    # Vertical road
    highway_v = patches.Rectangle((4, 0), 1.5, 10, linewidth=1, edgecolor='black', 
                               facecolor='#EEEEEE', zorder=1)
    ax_flow.add_patch(highway_v)
    
    # Lane markings
    for x in range(0, 10, 2):
        line_h = patches.Rectangle((x, 4.75), 0.8, 0.05, linewidth=0, facecolor='white', zorder=2)
        ax_flow.add_patch(line_h)
    
    for y in range(0, 10, 2):
        if y != 4 and y != 6:
            line_v = patches.Rectangle((4.75, y), 0.05, 0.8, linewidth=0, facecolor='white', zorder=2)
            ax_flow.add_patch(line_v)
    
    # Add traffic light
    traffic_light = patches.Circle((4.75, 4.75), 0.3, linewidth=1, edgecolor='black', 
                                facecolor='#4CAF50', zorder=5)
    ax_flow.add_patch(traffic_light)
    
    # Add many small vehicles to represent traffic flow
    car_colors = ['#1976D2', '#388E3C', '#E64A19', '#FBC02D', '#7B1FA2']
    
    # Horizontal traffic
    for i in range(7):
        x = i * 1.2 + 0.5
        if 3.5 < x < 6:  # Skip intersection
            continue
        car = patches.Rectangle((x, 4.3), 0.4, 0.2, linewidth=1, edgecolor='black', 
                             facecolor=np.random.choice(car_colors), zorder=3)
        ax_flow.add_patch(car)
    
    # Vertical traffic
    for i in range(5):
        y = i * 1.2 + 0.5
        if 3.5 < y < 6:  # Skip intersection
            continue
        car = patches.Rectangle((4.4, y), 0.2, 0.4, linewidth=1, edgecolor='black', 
                             facecolor=np.random.choice(car_colors), zorder=3)
        ax_flow.add_patch(car)
    
    # Control area visualization
    control_area = patches.Ellipse((4.75, 4.75), 6, 6, linewidth=1, edgecolor='#673AB7', 
                                 facecolor='none', linestyle='--', zorder=2)
    ax_flow.add_patch(control_area)
    
    # Draw arrows showing traffic flow optimization
    flow_arrow1 = patches.FancyArrowPatch((4.75, 7), (4.75, 6), arrowstyle='->', color='#673AB7', 
                                        linewidth=1.5, connectionstyle='arc3,rad=0', zorder=4)
    flow_arrow2 = patches.FancyArrowPatch((7, 4.75), (6, 4.75), arrowstyle='->', color='#673AB7', 
                                        linewidth=1.5, connectionstyle='arc3,rad=0', zorder=4)
    
    ax_flow.add_patch(flow_arrow1)
    ax_flow.add_patch(flow_arrow2)
    
    # Annotations
    ax_flow.text(4.75, 9, "Traffic Control Area", fontsize=10, ha='center', 
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='#673AB7'))
    
    # Key Flow features
    features_flow = [
        "Coordinated signal control",
        "Bottleneck prevention",
        "System-level optimization"
    ]
    
    for i, feature in enumerate(features_flow):
        ax_flow.text(6, 8.5 - i*0.6, f"• {feature}", fontsize=10, ha='left')
    
    # Add subtitle about traffic simulation benefits
    fig.text(0.5, 0.02, 
           "Traffic simulation platforms enable researchers to train and validate RL policies for autonomous driving and traffic control.",
           ha='center', va='bottom', fontsize=12, fontstyle='italic')
    
    # Save the figure
    output_path = os.path.join(output_dir, 'traffic_simulation.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust for suptitle and subtitle
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Traffic simulation visualization saved to {output_path}")

if __name__ == "__main__":
    create_isaac_gym_visualization()
    create_metaworld_visualization()
    create_traffic_simulation_visualization()
    print("Modern simulation visualizations complete!")
