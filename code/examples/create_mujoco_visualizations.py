#!/usr/bin/env python3

"""
This script creates visualizations for the MuJoCo Physics section:
1. Comparison diagram of different MuJoCo environments (Ant, HalfCheetah, Humanoid)
2. Learning curves for different algorithms on MuJoCo tasks
3. Visualization of contact dynamics and joint constraints
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_mujoco_environments_comparison():
    """
    Create a comparison visualization of different MuJoCo environments,
    showing their complexity, degrees of freedom, and state/action dimensions.
    """
    # Define environment data
    environments = [
        "Ant", "HalfCheetah", "Hopper", "Walker2d", "Humanoid", "HumanoidStandup"
    ]
    
    # Define properties for comparison
    state_dims = [111, 17, 11, 17, 376, 376]  # State dimensions
    action_dims = [8, 6, 3, 6, 17, 17]        # Action dimensions
    dof = [8, 6, 3, 6, 17, 17]                # Degrees of freedom
    reward_complexity = [2, 2, 3, 3, 5, 4]    # Subjective measure (1-5)
    learn_difficulty = [3, 2, 2, 3, 5, 5]     # Subjective measure (1-5)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1.5, 1])
    
    # Create bar chart for dimensions
    ax1 = plt.subplot(gs[0, 0])
    x = np.arange(len(environments))
    width = 0.35
    
    ax1.bar(x - width/2, state_dims, width, label='State Dimensions', color='#3498db', alpha=0.7)
    ax1.bar(x + width/2, action_dims, width, label='Action Dimensions', color='#e74c3c', alpha=0.7)
    
    ax1.set_ylabel('Dimensions', fontsize=12)
    ax1.set_title('State and Action Space Complexity', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(environments)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Create difficulty/complexity radar chart
    ax2 = plt.subplot(gs[0, 1], polar=True)
    
    # Number of variables
    categories = ['Learning\nDifficulty', 'Reward\nComplexity', 'Degrees of\nFreedom']
    N = len(categories)
    
    # We need to repeat the first value to close the circular graph
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Draw the polygons for each environment
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    
    for i, env in enumerate(environments):
        # Normalize values for radar chart
        values = [
            learn_difficulty[i] / 5.0,
            reward_complexity[i] / 5.0,
            dof[i] / max(dof)
        ]
        values += values[:1]
        
        ax2.plot(angles, values, color=colors[i], linewidth=2, label=env)
        ax2.fill(angles, values, color=colors[i], alpha=0.1)
    
    # Set category labels
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_title('Environment Characteristics', fontsize=14)
    
    # Create environment illustrations
    ax3 = plt.subplot(gs[1, :])
    
    # Environment sketches (simplified illustrations)
    # We'll create simple visualizations of the robots
    
    # Define positions for each environment illustration
    positions = [(0.1, 0.5), (0.28, 0.5), (0.46, 0.5), (0.64, 0.5), (0.82, 0.5)]
    
    # Draw simplified illustrations
    for i, env in enumerate(environments[:5]):  # Limit to 5 for space
        x, y = positions[i]
        
        if env == "Ant":
            # Draw ant-like body with 8 legs
            body = plt.Circle((x, y), 0.03, color=colors[i], alpha=0.8, zorder=10)
            ax3.add_patch(body)
            
            # Add legs
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                leg_x = x + 0.05 * np.cos(angle)
                leg_y = y + 0.05 * np.sin(angle)
                ax3.plot([x, leg_x], [y, leg_y], color=colors[i], lw=2, zorder=5)
                
            # Add label
            ax3.text(x, y - 0.12, env, ha='center', fontsize=12, fontweight='bold')
            
        elif env == "HalfCheetah":
            # Draw cheetah-like body
            body_x = [x-0.06, x+0.06]
            body_y = [y, y]
            ax3.plot(body_x, body_y, color=colors[i], lw=4, zorder=5)
            
            # Add legs
            ax3.plot([x-0.04, x-0.04, x-0.02], [y, y-0.05, y-0.08], color=colors[i], lw=2, zorder=5)
            ax3.plot([x+0.02, x+0.02, x+0.04], [y, y-0.05, y-0.08], color=colors[i], lw=2, zorder=5)
            
            # Add head
            head = plt.Circle((x+0.06, y), 0.015, color=colors[i], alpha=0.8, zorder=10)
            ax3.add_patch(head)
            
            # Add label
            ax3.text(x, y - 0.12, env, ha='center', fontsize=12, fontweight='bold')
            
        elif env == "Hopper":
            # Draw single-leg hopper
            body = plt.Circle((x, y+0.03), 0.025, color=colors[i], alpha=0.8, zorder=10)
            ax3.add_patch(body)
            
            # Add leg
            ax3.plot([x, x, x], [y+0.01, y-0.03, y-0.08], color=colors[i], lw=2, zorder=5)
            
            # Add label
            ax3.text(x, y - 0.12, env, ha='center', fontsize=12, fontweight='bold')
            
        elif env == "Walker2d":
            # Draw 2D walker
            body = plt.Rectangle((x-0.02, y+0.01), 0.04, 0.03, color=colors[i], alpha=0.8, zorder=10)
            ax3.add_patch(body)
            
            # Add legs
            ax3.plot([x-0.01, x-0.01, x-0.02], [y+0.01, y-0.04, y-0.08], color=colors[i], lw=2, zorder=5)
            ax3.plot([x+0.01, x+0.01, x+0.02], [y+0.01, y-0.04, y-0.08], color=colors[i], lw=2, zorder=5)
            
            # Add label
            ax3.text(x, y - 0.12, env, ha='center', fontsize=12, fontweight='bold')
            
        elif env == "Humanoid":
            # Draw humanoid
            body = plt.Rectangle((x-0.015, y+0.01), 0.03, 0.04, color=colors[i], alpha=0.8, zorder=10)
            ax3.add_patch(body)
            
            # Add head
            head = plt.Circle((x, y+0.06), 0.015, color=colors[i], alpha=0.8, zorder=10)
            ax3.add_patch(head)
            
            # Add limbs
            ax3.plot([x-0.015, x-0.03], [y+0.03, y+0.05], color=colors[i], lw=2, zorder=5)  # Left arm
            ax3.plot([x+0.015, x+0.03], [y+0.03, y+0.05], color=colors[i], lw=2, zorder=5)  # Right arm
            ax3.plot([x-0.01, x-0.01, x-0.02], [y+0.01, y-0.04, y-0.08], color=colors[i], lw=2, zorder=5)  # Left leg
            ax3.plot([x+0.01, x+0.01, x+0.02], [y+0.01, y-0.04, y-0.08], color=colors[i], lw=2, zorder=5)  # Right leg
            
            # Add label
            ax3.text(x, y - 0.12, env, ha='center', fontsize=12, fontweight='bold')
    
    ax3.axis('equal')
    ax3.axis('off')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0.3, 0.7)
    ax3.set_title('Simplified Environment Visualizations', fontsize=14)
    
    # Add title and adjust layout
    plt.suptitle('MuJoCo Environments Comparison', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'mujoco_environments_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"MuJoCo environments comparison saved to {os.path.join(output_dir, 'mujoco_environments_comparison.png')}")
    plt.close()

def create_learning_curves_comparison():
    """
    Create a visualization comparing learning curves of different algorithms
    on MuJoCo environments.
    """
    # Define algorithms and environments
    algorithms = ['PPO', 'SAC', 'TD3', 'DDPG']
    environments = ['HalfCheetah', 'Ant', 'Humanoid']
    
    # Algorithm colors
    colors = {
        'PPO': '#3498db',
        'SAC': '#e74c3c',
        'TD3': '#2ecc71',
        'DDPG': '#9b59b6'
    }
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Simulated learning curves data for each environment and algorithm
    # In practice, this would come from actual training runs
    timesteps = np.linspace(0, 1e6, 100)
    
    for i, env in enumerate(environments):
        ax = axes[i]
        
        # Different performance curves for each environment
        for alg in algorithms:
            # Simulate data with different characteristics for each algorithm
            if alg == 'PPO':
                # PPO: Steady but slower learning
                performance = 1000 * (1 - np.exp(-3 * timesteps / 1e6)) + np.random.normal(0, 50, size=len(timesteps))
                if env == 'Humanoid':
                    performance *= 0.7  # PPO struggles more with humanoid
                
            elif alg == 'SAC':
                # SAC: Fast initial progress with high final performance
                performance = 1200 * (1 - np.exp(-5 * timesteps / 1e6)) + np.random.normal(0, 40, size=len(timesteps))
                
            elif alg == 'TD3':
                # TD3: Similar to SAC but with different noise profile
                performance = 1100 * (1 - np.exp(-4 * timesteps / 1e6)) + np.random.normal(0, 60, size=len(timesteps))
                if env == 'Ant':
                    performance *= 1.1  # TD3 does well on Ant
                    
            elif alg == 'DDPG':
                # DDPG: More volatile learning curve
                performance = 900 * (1 - np.exp(-3.5 * timesteps / 1e6)) + np.random.normal(0, 80, size=len(timesteps))
                if env == 'Humanoid':
                    performance *= 0.6  # DDPG struggles with Humanoid
            
            # Scale based on environment (Humanoid has higher potential returns)
            if env == 'Humanoid':
                performance *= 1.5
            elif env == 'Ant':
                performance *= 1.2
            
            # Plot mean curve
            ax.plot(timesteps, performance, label=alg, color=colors[alg], linewidth=2)
            
            # Add shaded area for confidence interval
            std_dev = performance * 0.1  # 10% of mean as std dev
            ax.fill_between(timesteps, performance - std_dev, performance + std_dev, 
                           color=colors[alg], alpha=0.2)
        
        ax.set_title(f'{env}', fontsize=14)
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.grid(linestyle='--', alpha=0.3)
        
        # Format x-axis to show millions
        ax.ticklabel_format(axis='x', style='sci', scilimits=(6,6))
        
        # Only add y-label to leftmost plot
        if i == 0:
            ax.set_ylabel('Average Return', fontsize=12)
    
    # Add single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
              ncol=len(algorithms), fontsize=12)
    
    plt.suptitle('Algorithm Performance on MuJoCo Environments', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'mujoco_learning_curves.png'), dpi=300, bbox_inches='tight')
    print(f"MuJoCo learning curves saved to {os.path.join(output_dir, 'mujoco_learning_curves.png')}")
    plt.close()

def create_contact_dynamics_visualization():
    """
    Create a visualization of MuJoCo's contact dynamics and joint constraints.
    """
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Create 3D axes for contact visualization
    ax1 = plt.subplot(gs[0], projection='3d')
    
    # Create a simple visualization of contact dynamics
    # Ground plane
    xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    zz = np.zeros_like(xx)
    ax1.plot_surface(xx, yy, zz, color='gray', alpha=0.3)
    
    # Create a spherical object that will contact the ground
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    radius = 0.3
    
    # Calculate multiple positions of the sphere for animation-like effect
    heights = [1.0, 0.7, 0.4, 0.3, 0.3]  # Heights of sphere center
    deformations = [0.0, 0.0, 0.0, 0.05, 0.07]  # Deformation amount
    colors = ['#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c']  # Colors (blue->red when contact)
    
    for height, deform, color in zip(heights, deformations, colors):
        # Sphere coordinates
        x = radius * np.cos(u) * np.sin(v)
        y = radius * np.sin(u) * np.sin(v)
        
        # Apply deformation to the bottom of the sphere when contacting
        z = radius * np.cos(v)
        mask = z < -0.1  # Deform only the bottom part
        z[mask] = z[mask] * (1 - deform)
        
        # Translate to the right height
        z = z + height
        
        # Plot with transparency to see all frames
        ax1.plot_surface(x, y, z, color=color, alpha=0.3)
        
        # If in contact, draw contact points
        if deform > 0:
            contact_z = height - radius * (1 - deform)
            theta = np.linspace(0, 2*np.pi, 20)
            contact_x = 0.2 * np.cos(theta)
            contact_y = 0.2 * np.sin(theta)
            contact_points = np.zeros_like(contact_x)
            ax1.plot(contact_x, contact_y, contact_points + 0.01, 'ro', alpha=0.7)  # Contact points
            
            # Force vectors at contact points
            forces_x = np.zeros_like(contact_x)
            forces_y = np.zeros_like(contact_y)
            forces_z = np.ones_like(contact_x) * 0.2  # Upward force - Fixed to be array instead of scalar
            
            # Plot force vectors
            for i in range(len(contact_x)):
                if i % 3 == 0:  # Plot every 3rd point to avoid clutter
                    ax1.quiver(contact_x[i], contact_y[i], 0.01, 
                              forces_x[i], forces_y[i], forces_z[i],
                              color='red', alpha=0.7)
    
    ax1.set_title('Contact Dynamics in MuJoCo', fontsize=14)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(0, 2)
    ax1.view_init(elev=30, azim=45)
    
    # Create 2D axes for joint constraints visualization
    ax2 = plt.subplot(gs[1])
    
    # Draw a robotic arm with different joint types
    # Base of the arm
    base = plt.Rectangle((-0.6, -0.1), 1.2, 0.2, color='gray', alpha=0.7)
    ax2.add_patch(base)
    
    # Draw different joints and links
    # Joint 1: Revolute (hinge) joint - full 360° rotation
    joint1_center = (0, 0.2)
    joint1 = plt.Circle(joint1_center, 0.1, color='#3498db', alpha=0.8)
    ax2.add_patch(joint1)
    
    # Show rotation range with an arc
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(joint1_center[0] + 0.15 * np.cos(theta), 
             joint1_center[1] + 0.15 * np.sin(theta), 
             color='#3498db', linestyle='--', alpha=0.7)
    
    # Link 1
    link1_end = (0, 0.8)
    ax2.plot([joint1_center[0], link1_end[0]], [joint1_center[1], link1_end[1]], 
             color='black', lw=3)
    
    # Joint 2: Limited hinge joint
    joint2_center = link1_end
    joint2 = plt.Circle(joint2_center, 0.1, color='#e74c3c', alpha=0.8)
    ax2.add_patch(joint2)
    
    # Show limited rotation range with partial arc
    theta = np.linspace(-np.pi/4, np.pi/4, 100)  # Limited range
    ax2.plot(joint2_center[0] + 0.15 * np.cos(theta), 
             joint2_center[1] + 0.15 * np.sin(theta), 
             color='#e74c3c', linestyle='--', alpha=0.7)
    
    # Link 2
    link2_end = (0.4, 1.2)
    ax2.plot([joint2_center[0], link2_end[0]], [joint2_center[1], link2_end[1]], 
             color='black', lw=3)
    
    # Joint 3: Ball joint (spherical)
    joint3_center = link2_end
    joint3 = plt.Circle(joint3_center, 0.1, color='#2ecc71', alpha=0.8)
    ax2.add_patch(joint3)
    
    # Show 3D rotation with multiple rings
    for angle in [0, np.pi/4, np.pi/2]:
        a = 0.15 * np.cos(angle)
        b = 0.15 * np.sin(angle)
        if angle == 0:
            theta = np.linspace(0, 2*np.pi, 100)
            ax2.plot(joint3_center[0] + a * np.cos(theta), 
                    joint3_center[1] + b * np.sin(theta), 
                    color='#2ecc71', linestyle='--', alpha=0.7)
        else:
            theta = np.linspace(0, 2*np.pi, 100)
            x = joint3_center[0] + a * np.cos(theta)
            y = joint3_center[1] + b * np.sin(theta)
            ax2.plot(x, y, color='#2ecc71', linestyle='--', alpha=0.7)
    
    # Link 3 (end effector)
    link3_end = (0.7, 1.5)
    ax2.plot([joint3_center[0], link3_end[0]], [joint3_center[1], link3_end[1]], 
             color='black', lw=3)
    
    # Annotations
    ax2.annotate('Revolute Joint\n(360° rotation)', xy=joint1_center, xytext=(-0.5, 0.2),
                arrowprops=dict(arrowstyle='->'), fontsize=10)
    
    ax2.annotate('Limited Hinge Joint\n(±45° range)', xy=joint2_center, xytext=(-0.7, 0.8),
                arrowprops=dict(arrowstyle='->'), fontsize=10)
    
    ax2.annotate('Ball Joint\n(spherical motion)', xy=joint3_center, xytext=(0.7, 0.9),
                arrowprops=dict(arrowstyle='->'), fontsize=10)
    
    ax2.set_title('Joint Types and Constraints', fontsize=14)
    ax2.set_xlim(-1, 1.5)
    ax2.set_ylim(-0.2, 1.7)
    ax2.axis('equal')
    ax2.axis('off')
    
    # Adjust layout and save
    plt.suptitle('MuJoCo Physics Simulation Features', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'mujoco_physics_features.png'), dpi=300, bbox_inches='tight')
    print(f"MuJoCo physics features visualization saved to {os.path.join(output_dir, 'mujoco_physics_features.png')}")
    plt.close()

if __name__ == "__main__":
    print("Generating MuJoCo environment comparisons...")
    create_mujoco_environments_comparison()
    
    print("Generating learning curves comparison...")
    create_learning_curves_comparison()
    
    print("Generating contact dynamics visualization...")
    create_contact_dynamics_visualization()
    
    print("All MuJoCo visualizations complete!") 