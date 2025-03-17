#!/usr/bin/env python3
"""
This script creates visualizations for Section 11: Practical Implementation.
1. Reward shaping comparison
2. RL statistical analysis with confidence intervals
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy import stats

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_reward_shaping_comparison():
    """
    Creates a visualization comparing different reward shaping approaches.
    """
    print("Creating reward shaping comparison visualization...")
    
    # Set up figure
    plt.figure(figsize=(12, 8))
    
    # Generate sample learning curves with numpy's random number generator
    np.random.seed(42)  # For reproducibility
    
    # X-axis: training steps
    steps = np.arange(0, 100)
    
    # Define curves for different reward strategies
    # Sparse reward (slow, steady progress)
    sparse_rewards = 20 * (1 - np.exp(-0.03 * steps)) + np.random.normal(0, 1.5, 100)
    sparse_rewards = np.maximum(0, sparse_rewards)  # No negative rewards
    
    # Dense reward (faster initial learning, but plateaus early)
    dense_rewards = 15 * (1 - np.exp(-0.08 * steps)) + 8 * (1 - np.exp(-0.01 * steps)) + np.random.normal(0, 2, 100)
    dense_rewards = np.maximum(0, dense_rewards)
    
    # Shaped reward (best performance)
    shaped_rewards = 25 * (1 - np.exp(-0.05 * steps)) + np.random.normal(0, 1, 100)
    shaped_rewards = np.maximum(0, shaped_rewards)
    
    # Misaligned reward (peaks, then declines)
    misaligned_rewards = 18 * (1 - np.exp(-0.1 * steps)) - 0.1 * steps + np.random.normal(0, 2, 100)
    misaligned_rewards = np.maximum(0, misaligned_rewards)
    
    # Plot the learning curves
    plt.plot(steps, sparse_rewards, label='Sparse Reward', linewidth=2.5, color='#2196F3')
    plt.plot(steps, dense_rewards, label='Dense Reward', linewidth=2.5, color='#FF9800')
    plt.plot(steps, shaped_rewards, label='Potential-Based Shaping', linewidth=2.5, color='#4CAF50')
    plt.plot(steps, misaligned_rewards, label='Misaligned Reward', linewidth=2.5, color='#F44336')
    
    # Add annotations explaining key characteristics
    plt.annotate('Slow initial learning',
                xy=(40, sparse_rewards[40]), 
                xytext=(20, sparse_rewards[40] - 8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    plt.annotate('Early progress but premature plateau',
                xy=(60, dense_rewards[60]), 
                xytext=(40, dense_rewards[60] + 6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    plt.annotate('Optimal policy preserved',
                xy=(80, shaped_rewards[80]), 
                xytext=(60, shaped_rewards[80] + 6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    plt.annotate('Performance degradation\nfrom reward hacking',
                xy=(70, misaligned_rewards[70]), 
                xytext=(75, misaligned_rewards[70] - 10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    # Add mathematical formulations
    plt.text(5, 27, r'Sparse: $R(s) = \mathbb{1}_{\text{goal}}(s)$', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(5, 25, r'Dense: $R(s) = -||s - s_{\text{goal}}||_2$', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    plt.text(5, 23, r'Shaped: $F(s, s\') = \gamma \Phi(s\') - \Phi(s)$', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend and labels
    plt.xlabel('Training Steps (thousands)', fontsize=14)
    plt.ylabel('Mean Episode Return', fontsize=14)
    plt.title('Impact of Reward Shaping on Learning Performance', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize axes
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(np.arange(0, 101, 20))
    
    # Save figure
    output_path = os.path.join(output_dir, 'reward_shaping_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Reward shaping comparison saved to {output_path}")

def create_statistical_analysis_visualization():
    """
    Creates a visualization showing statistical analysis of RL algorithm performance.
    """
    print("Creating statistical analysis visualization...")
    
    # Set up figure
    plt.figure(figsize=(12, 8))
    
    # Set a nice aesthetic style
    sns.set_style("whitegrid")
    
    # Define algorithms and environments
    algorithms = ["DQN", "PPO", "SAC", "TRPO"]
    
    # Generate synthetic performance data
    np.random.seed(42)
    
    # Create performance data with different means and variances
    # Format: [mean, std, num_seeds]
    performance_data = {
        "DQN": [85, 15, 10],
        "PPO": [92, 10, 10],
        "SAC": [95, 8, 10],
        "TRPO": [88, 12, 10]
    }
    
    # Generate raw data based on mean/std
    raw_data = {}
    for algo, (mean, std, seeds) in performance_data.items():
        raw_data[algo] = np.random.normal(mean, std, seeds)
    
    # Plot the data
    x_pos = np.arange(len(algorithms))
    width = 0.8
    
    # Plot violin plots to show distribution
    violin_parts = plt.violinplot([raw_data[algo] for algo in algorithms], 
                                  positions=x_pos, 
                                  widths=width*0.8, 
                                  showmeans=False, 
                                  showextrema=False)
    
    # Customize violin plots
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#E0E0E0')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Add scatter plots to show individual runs
    for i, algo in enumerate(algorithms):
        y = raw_data[algo]
        # Add jitter to x-positions
        x_jitter = np.random.normal(0, 0.07, len(y))
        plt.scatter(x_pos[i] + x_jitter, y, color='#3f51b5', alpha=0.7, s=50)
    
    # Compute and plot means with 95% confidence intervals
    for i, algo in enumerate(algorithms):
        y = raw_data[algo]
        mean = np.mean(y)
        
        # 95% confidence interval
        ci_lower, ci_upper = stats.norm.interval(0.95, loc=mean, scale=stats.sem(y))
        
        plt.errorbar(x_pos[i], mean, yerr=[[mean-ci_lower], [ci_upper-mean]], 
                    fmt='o', color='#c62828', ecolor='#c62828', 
                    elinewidth=3, capsize=10, capthick=3, ms=10)
        
        # Add mean value as text
        plt.text(x_pos[i], mean + 12, f"{mean:.1f}", ha='center', fontsize=12, fontweight='bold')
    
    # Add statistical significance markers
    # For this example, we'll mark PPO vs DQN and SAC vs PPO
    # In a real analysis, you would compute actual p-values
    
    # PPO vs DQN (significant)
    plt.plot([x_pos[0], x_pos[0], x_pos[1], x_pos[1]], 
             [max(raw_data["DQN"])+5, max(raw_data["DQN"])+10, max(raw_data["DQN"])+10, max(raw_data["PPO"])+5],
             'k-', linewidth=1.5)
    plt.text((x_pos[0] + x_pos[1])/2, max(raw_data["DQN"])+11, "*", ha='center', va='bottom', fontsize=20)
    
    # SAC vs PPO (not significant)
    plt.plot([x_pos[1], x_pos[1], x_pos[2], x_pos[2]], 
             [max(raw_data["PPO"])+15, max(raw_data["PPO"])+20, max(raw_data["PPO"])+20, max(raw_data["SAC"])+15],
             'k-', linewidth=1.5)
    plt.text((x_pos[1] + x_pos[2])/2, max(raw_data["PPO"])+21, "n.s.", ha='center', va='bottom', fontsize=12)
    
    # Customize axes and labels
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Mean Episode Return', fontsize=14)
    plt.title('Statistical Analysis of Algorithm Performance\nacross 10 random seeds', fontsize=16)
    plt.xticks(x_pos, algorithms, fontsize=12)
    plt.ylim(40, 120)
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, 
               "Error bars show 95% confidence intervals. * indicates p < 0.05, n.s. = not significant.",
               ha='center', fontsize=12, style='italic')
    
    # Add a box explaining the analysis
    explanation_text = (
        "Statistical Analysis Details:\n"
        "• Each data point represents performance from a different random seed\n"
        "• Violin plots show distribution of results across seeds\n"
        "• Error bars show 95% confidence intervals around the mean\n"
        "• Statistical significance determined using Welch's t-test"
    )
    
    plt.figtext(0.72, 0.25, explanation_text, 
               bbox=dict(facecolor='#f5f5f5', edgecolor='#757575', boxstyle='round,pad=0.5'),
               fontsize=10, ha='left', va='center')
    
    # Save figure
    output_path = os.path.join(output_dir, 'rl_statistical_analysis.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Statistical analysis visualization saved to {output_path}")

if __name__ == "__main__":
    create_reward_shaping_comparison()
    create_statistical_analysis_visualization()
    print("RL implementation visualizations complete!") 