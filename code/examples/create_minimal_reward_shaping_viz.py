#!/usr/bin/env python3
"""
Creates a minimalist visualization of reward shaping impact on learning performance
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_minimal_reward_shaping_viz():
    """
    Creates a clean, minimalist visualization comparing different reward shaping approaches.
    """
    print("Creating minimalist reward shaping visualization...")
    
    # Set figure size and style
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Use specific random seed for reproducibility
    np.random.seed(42)
    
    # Create more realistic learning curves with 500 episodes
    episodes = np.arange(0, 501)
    
    # Function to create smoothed, realistic learning curves with noise
    def learning_curve(start_val, end_val, speed, noise_level, plateau_point=None, decline_rate=0):
        # Base learning curve shape
        curve = start_val + (end_val - start_val) * (1 - np.exp(-speed * episodes / 100))
        
        # Add plateau effect
        if plateau_point:
            plateau_mask = episodes > plateau_point
            curve[plateau_mask] = curve[plateau_point] + (curve[plateau_mask] - curve[plateau_point]) * 0.3
        
        # Add decline effect
        if decline_rate > 0:
            decline_mask = episodes > 200
            curve[decline_mask] -= decline_rate * (episodes[decline_mask] - 200) / 10
        
        # Add noise with smoothing
        noise = np.random.normal(0, noise_level, len(episodes))
        # Smoothing with rolling average
        window_size = 5
        smoothed_noise = np.convolve(noise, np.ones(window_size)/window_size, mode='same')
        
        return np.maximum(0, curve + smoothed_noise)
    
    # Generate realistic curves
    sparse_reward = learning_curve(5, 190, 0.6, 3.0)  # Slow but eventually reaches good performance
    dense_reward = learning_curve(15, 190, 1.5, 4.0, plateau_point=300)  # Faster initial learning, plateaus
    potential_based = learning_curve(10, 200, 1.2, 2.0)  # Consistent improvement to optimal
    poorly_designed = learning_curve(20, 150, 2.5, 5.0, plateau_point=150, decline_rate=0.4)  # Fast start but unstable and suboptimal
    
    # Plot the learning curves with clearer colors and thicker lines
    plt.plot(episodes, sparse_reward, label='Sparse Reward', linewidth=2.5, color='#1976D2')  # Blue
    plt.plot(episodes, dense_reward, label='Dense Reward', linewidth=2.5, color='#FF9800')  # Orange
    plt.plot(episodes, potential_based, label='Potential-Based Shaping', linewidth=2.5, color='#4CAF50')  # Green
    plt.plot(episodes, poorly_designed, label='Poorly Designed Reward', linewidth=2.5, color='#E53935')  # Red
    
    # Add optimal performance line
    plt.axhline(y=200, linestyle='--', color='#9E9E9E', alpha=0.7, linewidth=1.5, label='Optimal Performance')
    
    # Add minimal annotations - only the most essential
    plt.text(450, 205, "Optimal", fontsize=11, color='#616161', ha='right')
    plt.text(400, 170, "Potential-Based", fontsize=11, color='#4CAF50', ha='center')
    plt.text(400, 140, "Poorly Designed", fontsize=11, color='#E53935', ha='center')
    
    # Clean, minimal axes and labels
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.title('Impact of Reward Shaping on Learning Performance', fontsize=16, pad=20)
    
    # Set y-axis to start from 0
    plt.ylim(0, 220)
    plt.xlim(0, 500)
    
    # Add minimal legend and remove box around it
    plt.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=12)
    
    # Remove unnecessary spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Save figure with tight layout
    output_path = os.path.join(output_dir, 'minimal_reward_shaping.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Minimalist reward shaping visualization saved to {output_path}")

if __name__ == "__main__":
    create_minimal_reward_shaping_viz()
    print("Visualization complete!") 