#!/usr/bin/env python3
"""
Creates a minimalist visualization of statistical analysis for RL experiments
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_minimal_statistical_analysis_viz():
    """
    Creates a clean, minimalist visualization of statistical analysis
    for reinforcement learning experiments.
    """
    print("Creating minimalist statistical analysis visualization...")
    
    # Use specific random seed for reproducibility
    np.random.seed(42)
    
    # Create the figure with a clean white background
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    
    # Use a simple 2x1 layout - focusing only on the most essential panels
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5])
    
    # ---- Panel 1: Learning Curves with Confidence Intervals ----
    ax1 = plt.subplot(gs[0])
    
    # Create timesteps
    timesteps = np.arange(0, 250)
    
    # Function to create smoothed learning curves
    def create_curve(base_values, noise_level=10, smoothing_window=10):
        # Add noise
        noise = np.random.normal(0, noise_level, len(base_values))
        # Apply smoothing
        curve = base_values + noise
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            curve = np.convolve(curve, kernel, mode='same')
        return curve
    
    # Generate baseline curves
    algorithm_a_base = 20 + 150 * (1 - np.exp(-0.015 * timesteps))
    algorithm_b_base = 30 + 160 * (1 - np.exp(-0.018 * timesteps))
    algorithm_c_base = 10 + 140 * (1 - np.exp(-0.012 * timesteps))
    
    # Generate multiple seeds for confidence intervals
    n_seeds = 10
    
    # Algorithm A (blue)
    a_curves = np.array([create_curve(algorithm_a_base, noise_level=15) for _ in range(n_seeds)])
    a_mean = np.mean(a_curves, axis=0)
    a_std = np.std(a_curves, axis=0)
    a_ci_lower = a_mean - 1.96 * a_std / np.sqrt(n_seeds)
    a_ci_upper = a_mean + 1.96 * a_std / np.sqrt(n_seeds)
    
    # Algorithm B (green)
    b_curves = np.array([create_curve(algorithm_b_base, noise_level=20) for _ in range(n_seeds)])
    b_mean = np.mean(b_curves, axis=0)
    b_std = np.std(b_curves, axis=0)
    b_ci_lower = b_mean - 1.96 * b_std / np.sqrt(n_seeds)
    b_ci_upper = b_mean + 1.96 * b_std / np.sqrt(n_seeds)
    
    # Algorithm C (orange)
    c_curves = np.array([create_curve(algorithm_c_base, noise_level=12) for _ in range(n_seeds)])
    c_mean = np.mean(c_curves, axis=0)
    c_std = np.std(c_curves, axis=0)
    c_ci_lower = c_mean - 1.96 * c_std / np.sqrt(n_seeds)
    c_ci_upper = c_mean + 1.96 * c_std / np.sqrt(n_seeds)
    
    # Plot confidence intervals as shaded areas
    ax1.fill_between(timesteps, a_ci_lower, a_ci_upper, alpha=0.2, color='#1976D2')
    ax1.fill_between(timesteps, b_ci_lower, b_ci_upper, alpha=0.2, color='#4CAF50')
    ax1.fill_between(timesteps, c_ci_lower, c_ci_upper, alpha=0.2, color='#FF9800')
    
    # Plot mean curves
    ax1.plot(timesteps, a_mean, color='#1976D2', linewidth=2.5, label='Algorithm A')
    ax1.plot(timesteps, b_mean, color='#4CAF50', linewidth=2.5, label='Algorithm B')
    ax1.plot(timesteps, c_mean, color='#FF9800', linewidth=2.5, label='Algorithm C')
    
    # Minimal clean styling
    ax1.set_xlabel('Timesteps (thousands)', fontsize=12)
    ax1.set_ylabel('Average Return', fontsize=12)
    ax1.set_title('Learning Curves with 95% Confidence Intervals', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right')
    
    # Minimal annotation showing significance
    # Highlight two confidence interval areas - one overlapping, one not
    ax1.annotate(
        'Significant difference', 
        xy=(200, b_mean[200]), 
        xytext=(200, b_mean[200] + 30),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=10,
        ha='center'
    )
    
    ax1.annotate(
        'No significant difference', 
        xy=(200, a_mean[200]), 
        xytext=(200, a_mean[200] - 30),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=10,
        ha='center'
    )
    
    # Remove unnecessary spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ---- Panel 2: Algorithm Comparison with Statistical Significance ----
    ax2 = plt.subplot(gs[1])
    
    # Final performance values (simplified from previous panel)
    final_values = [
        ('A', a_mean[-1], a_std[-1]),
        ('B', b_mean[-1], b_std[-1]),
        ('C', c_mean[-1], c_std[-1]),
        ('D', 125, 15),  # Adding a fourth algorithm for comparison
        ('E', 165, 10)   # Adding a fifth algorithm for comparison
    ]
    
    # Sort by performance (descending)
    final_values.sort(key=lambda x: x[1], reverse=True)
    
    # Extract data for plotting
    algorithms = [x[0] for x in final_values]
    performance = [x[1] for x in final_values]
    errors = [1.96 * x[2] / np.sqrt(n_seeds) for x in final_values]  # 95% CI
    
    # Set colors based on significance
    colors = ['#1976D2', '#7B1FA2', '#4CAF50', '#FF9800', '#E53935']
    
    # Create bar chart
    bars = ax2.bar(algorithms, performance, yerr=errors, capsize=6, 
                   color=colors, alpha=0.8, width=0.6, edgecolor='black', linewidth=1)
    
    # Add minimal p-value annotations
    ax2.annotate('p=0.03', xy=(2, performance[2]), xytext=(2, performance[2] + 15), 
                 ha='center', va='bottom', fontsize=10)
    ax2.annotate('p<0.01', xy=(3, performance[3]), xytext=(3, performance[3] + 15), 
                 ha='center', va='bottom', fontsize=10)
    
    # Clean styling
    ax2.set_ylabel('Final Return', fontsize=12)
    ax2.set_title('Performance Comparison with Statistical Significance', fontsize=14)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Remove unnecessary spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add overall title
    plt.suptitle('Statistical Analysis in Reinforcement Learning', fontsize=16, y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_dir, 'minimal_statistical_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Minimalist statistical analysis visualization saved to {output_path}")

if __name__ == "__main__":
    create_minimal_statistical_analysis_viz()
    print("Visualization complete!") 