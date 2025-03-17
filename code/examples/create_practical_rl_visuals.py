#!/usr/bin/env python3
"""
This script creates visualizations for Section 11: Practical Implementation
- Reward shaping comparison
- Statistical analysis of RL experiments
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_reward_shaping_comparison():
    """
    Creates a visualization comparing different reward shaping strategies and their impact
    on learning speed and performance.
    """
    print("Creating reward shaping comparison visualization...")
    
    # Setup the figure
    plt.figure(figsize=(14, 8))
    plt.title("Impact of Reward Shaping on Learning Performance", fontsize=20, pad=15)
    
    # X-axis: episodes
    episodes = np.arange(0, 500)
    
    # Generate learning curves for different reward shaping strategies
    # Using sigmoid functions with different parameters to mimic learning curves
    
    # 1. Sparse reward (slow learning but eventually reaches optimal)
    def sparse_reward_curve(x):
        return 200 / (1 + np.exp(-0.015 * (x - 250)))
    
    # 2. Dense reward (faster initial learning but may converge to suboptimal solution)
    def dense_reward_curve(x):
        return 180 / (1 + np.exp(-0.025 * (x - 150))) + 10
    
    # 3. Potential-based shaping (fast learning with optimal convergence)
    def potential_based_curve(x):
        return 200 / (1 + np.exp(-0.03 * (x - 120)))
    
    # 4. Ill-designed reward (quick learning that plateaus suboptimally)
    def bad_reward_curve(x):
        return 120 / (1 + np.exp(-0.05 * (x - 80))) + 5 * np.sin(x/30) + 20
    
    # Add some noise to the curves
    rng = np.random.RandomState(42)
    
    # Generate the curves with noise
    sparse_rewards = sparse_reward_curve(episodes) + rng.normal(0, 10, size=len(episodes))
    dense_rewards = dense_reward_curve(episodes) + rng.normal(0, 8, size=len(episodes))
    potential_rewards = potential_based_curve(episodes) + rng.normal(0, 5, size=len(episodes))
    bad_rewards = bad_reward_curve(episodes) + rng.normal(0, 15, size=len(episodes))
    
    # Smooth the curves with rolling average
    window_size = 15
    
    def smooth(y):
        return np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    
    # Plot the curves
    plt.plot(episodes[window_size-1:], smooth(sparse_rewards), 
             label="Sparse Reward", color="#1f77b4", linewidth=3)
    plt.plot(episodes[window_size-1:], smooth(dense_rewards), 
             label="Dense Reward", color="#ff7f0e", linewidth=3)
    plt.plot(episodes[window_size-1:], smooth(potential_rewards), 
             label="Potential-Based Shaping", color="#2ca02c", linewidth=3)
    plt.plot(episodes[window_size-1:], smooth(bad_rewards), 
             label="Poorly Designed Reward", color="#d62728", linewidth=3)
    
    # Add a horizontal line for optimal performance
    plt.axhline(y=200, color='gray', linestyle='--', alpha=0.7, label="Optimal Performance")
    
    # Annotate key points
    plt.annotate("Slow initial learning", 
                xy=(150, sparse_reward_curve(150)), 
                xytext=(100, 60),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#1f77b4"),
                color="#1f77b4")
    
    plt.annotate("Faster but suboptimal", 
                xy=(300, dense_reward_curve(300)), 
                xytext=(320, 150),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color="#ff7f0e"),
                color="#ff7f0e")
    
    plt.annotate("Fast learning & optimal policy", 
                xy=(200, potential_based_curve(200)), 
                xytext=(150, 220),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color="#2ca02c"),
                color="#2ca02c")
    
    plt.annotate("Unstable learning", 
                xy=(350, bad_reward_curve(350)), 
                xytext=(370, 100),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#d62728"),
                color="#d62728")
    
    # Customize the plot
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Average Reward", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="lower right")
    
    # Add an explanatory text box
    text_str = """
    Potential-based reward shaping ensures:
    - Faster learning than sparse rewards
    - Same optimal policy as original task
    - Less bias than arbitrary dense rewards
    - Formal guarantee: F(s,s') = γΦ(s') - Φ(s)
    """
    
    plt.figtext(0.15, 0.15, text_str, bbox=dict(facecolor='white', alpha=0.8, 
                                               boxstyle='round,pad=0.5', edgecolor='#2ca02c'))
    
    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'reward_shaping_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Reward shaping comparison saved to {output_path}")

def create_statistical_analysis_visualization():
    """
    Creates a visualization showing statistical analysis of RL experiments,
    including confidence intervals and significance testing.
    """
    print("Creating statistical analysis visualization...")
    
    # Setup the figure with gridspec
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.3)
    
    # Add a big title for the entire figure
    fig.suptitle("Statistical Analysis in Reinforcement Learning Experiments", fontsize=20, y=0.98)
    
    # ---- Learning curves with confidence intervals (top-left) ----
    ax1 = plt.subplot(gs[0, 0])
    ax1.set_title("Learning Curves with 95% Confidence Intervals", fontsize=16)
    
    # X-axis: timesteps (in thousands)
    timesteps = np.arange(0, 250, 5)
    
    # Generate learning curves for different algorithms with confidence intervals
    def algorithm_curve(x, offset, scale, noise_scale):
        base_curve = scale / (1 + np.exp(-0.03 * (x - offset)))
        mean_curve = base_curve + np.sin(x/20) * 5
        
        rng = np.random.RandomState(42)
        curves = []
        for _ in range(10):  # Generate 10 random seeds
            noise = rng.normal(0, noise_scale, size=len(x))
            seed_curve = mean_curve + noise
            curves.append(seed_curve)
        
        curves = np.array(curves)
        mean = np.mean(curves, axis=0)
        lower = np.percentile(curves, 2.5, axis=0)  # 2.5th percentile for 95% CI
        upper = np.percentile(curves, 97.5, axis=0)  # 97.5th percentile for 95% CI
        
        return mean, lower, upper
    
    # Generate curves for three algorithms
    algo1_mean, algo1_lower, algo1_upper = algorithm_curve(timesteps, 80, 180, 15)
    algo2_mean, algo2_lower, algo2_upper = algorithm_curve(timesteps, 60, 200, 20)
    algo3_mean, algo3_lower, algo3_upper = algorithm_curve(timesteps, 100, 160, 10)
    
    # Plot the curves with confidence intervals
    ax1.plot(timesteps, algo1_mean, label="Algorithm A", color="#1f77b4", linewidth=2)
    ax1.fill_between(timesteps, algo1_lower, algo1_upper, color="#1f77b4", alpha=0.2)
    
    ax1.plot(timesteps, algo2_mean, label="Algorithm B", color="#2ca02c", linewidth=2)
    ax1.fill_between(timesteps, algo2_lower, algo2_upper, color="#2ca02c", alpha=0.2)
    
    ax1.plot(timesteps, algo3_mean, label="Algorithm C", color="#ff7f0e", linewidth=2)
    ax1.fill_between(timesteps, algo3_lower, algo3_upper, color="#ff7f0e", alpha=0.2)
    
    # Add an arrow and annotation pointing to overlapping confidence intervals
    ax1.annotate("Overlapping CIs:\nNo significant difference", 
                xy=(150, algo1_mean[30]), 
                xytext=(110, 90),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add an arrow and annotation pointing to non-overlapping confidence intervals
    ax1.annotate("Non-overlapping CIs:\nStatistically significant difference", 
                xy=(200, (algo2_mean[40] + algo3_mean[40])/2), 
                xytext=(180, 60),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Customize the plot
    ax1.set_xlabel("Timesteps (thousands)", fontsize=12)
    ax1.set_ylabel("Average Return", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="lower right")
    
    # ---- Final performance comparison (top-right) ----
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_title("Final Performance Comparison", fontsize=16)
    
    # Generate final performance data with error bars
    algorithms = ["A", "B", "C", "D", "E"]
    performance = [175, 195, 160, 130, 185]
    error = [15, 20, 10, 25, 18]  # 95% confidence intervals
    
    # Sort by performance
    sorted_indices = np.argsort(performance)[::-1]  # Descending
    algorithms = [algorithms[i] for i in sorted_indices]
    performance = [performance[i] for i in sorted_indices]
    error = [error[i] for i in sorted_indices]
    
    # Define colors based on statistical significance
    # Assuming B significantly outperforms A, while others aren't sig. different from neighbors
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    colors = [colors[i] for i in sorted_indices]
    
    # Plot the bar chart
    bars = ax2.bar(algorithms, performance, yerr=error, capsize=5, color=colors, alpha=0.7, width=0.6)
    
    # Add p-value annotations
    ax2.annotate("p < 0.01", xy=(0.5, 200), ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    ax2.annotate("p = 0.32", xy=(1.5, 180), ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    ax2.annotate("p = 0.03", xy=(2.5, 150), ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    ax2.annotate("p < 0.01", xy=(3.5, 135), ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Customize the plot
    ax2.set_ylabel("Final Return", fontsize=12)
    ax2.set_ylim(0, 250)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add a note about p-values
    ax2.text(0.5, 40, "p-values show statistical significance\nbetween adjacent algorithms", 
            ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # ---- Multiple seeds analysis (bottom-left) ----
    ax3 = plt.subplot(gs[1, 0])
    ax3.set_title("Impact of Seed Variability", fontsize=16)
    
    # Generate sample seeds data
    num_seeds = 10
    seed_performance = algo2_mean[-1] + np.random.RandomState(42).normal(0, 20, num_seeds)
    
    # Plot individual seed results
    ax3.bar(range(1, num_seeds+1), seed_performance, color="#2ca02c", alpha=0.5)
    
    # Add mean line
    mean_perf = np.mean(seed_performance)
    ax3.axhline(y=mean_perf, color='red', linestyle='-', linewidth=2, label=f"Mean: {mean_perf:.1f}")
    
    # Add confidence interval lines
    ci_low = np.percentile(seed_performance, 2.5)
    ci_high = np.percentile(seed_performance, 97.5)
    ax3.axhline(y=ci_low, color='blue', linestyle='--', linewidth=1.5, label=f"95% CI: [{ci_low:.1f}, {ci_high:.1f}]")
    ax3.axhline(y=ci_high, color='blue', linestyle='--', linewidth=1.5)
    
    # Fill the confidence interval area
    ax3.fill_between([0.5, num_seeds+0.5], ci_low, ci_high, color='blue', alpha=0.1)
    
    # Customize the plot
    ax3.set_xlabel("Random Seed", fontsize=12)
    ax3.set_ylabel("Final Performance", fontsize=12)
    ax3.set_xlim(0.5, num_seeds+0.5)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc="upper right")
    
    # ---- Significance testing workflow (bottom-right) ----
    ax4 = plt.subplot(gs[1, 1])
    ax4.set_title("Statistical Significance Workflow", fontsize=16)
    ax4.axis('off')
    
    # Create a flowchart-like diagram
    steps = [
        "1. Run multiple seeds (5-10 minimum)",
        "2. Compute mean and 95% CI",
        "3. Compare with baseline/other methods",
        "4. Apply statistical tests (e.g., t-test)",
        "5. Calculate effect size (Cohen's d)",
        "6. Report all metrics in publications"
    ]
    
    for i, step in enumerate(steps):
        y_pos = 0.85 - i * 0.15
        ax4.text(0.5, y_pos, step, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#aaaaaa", alpha=0.8))
        
        if i < len(steps) - 1:
            arrow_start = (0.5, y_pos - 0.06)
            arrow_end = (0.5, y_pos - 0.09)
            ax4.annotate("", xy=arrow_end, xytext=arrow_start, 
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # Add a final note
    ax4.text(0.5, 0.05, "Reproducible research requires\nrigorous statistical analysis", 
            ha='center', va='center', fontsize=10, fontstyle='italic',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.2))
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    output_path = os.path.join(output_dir, 'rl_statistical_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Statistical analysis visualization saved to {output_path}")

if __name__ == "__main__":
    print("Generating visualizations for Section 11: Practical Implementation...")
    create_reward_shaping_comparison()
    create_statistical_analysis_visualization()
    print("Visualizations complete!") 