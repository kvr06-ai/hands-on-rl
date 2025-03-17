#!/usr/bin/env python3
"""
This script creates GIF animations of classic control environments:
1. CartPole: balancing a pole on a cart
2. Mountain Car: an underpowered car trying to climb a hill
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Polygon
import os
from PIL import Image
import io

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_cartpole_gif(filename='cartpole.gif', num_frames=200, fps=30):
    """
    Create an animation of the CartPole environment.
    """
    # Set up the environment
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env.reset(seed=42)
    
    # Run a basic policy for demonstration
    frames = []
    observation, _ = env.reset()
    
    for _ in range(num_frames):
        # Use a simple heuristic: move cart in direction of pole's lean
        pole_angle = observation[2]
        action = 0 if pole_angle < 0 else 1
        
        # Take action and capture the rendered frame
        observation, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(Image.fromarray(frame))
        
        if terminated or truncated:
            observation, _ = env.reset()
    
    # Save animation as GIF
    frames[0].save(
        os.path.join(output_dir, filename),
        save_all=True,
        append_images=frames[1:],
        duration=1000/fps,
        loop=0
    )
    
    env.close()
    print(f"CartPole GIF saved to {os.path.join(output_dir, filename)}")

def create_mountain_car_gif(filename='mountain_car.gif', num_frames=200, fps=30):
    """
    Create an animation of the Mountain Car environment.
    """
    # Set up the environment
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    
    # Run a policy that shows oscillation behavior
    frames = []
    observation, _ = env.reset(seed=42)
    
    # Use a pre-trained policy or a decent heuristic
    # This is a simple oscillation heuristic
    for i in range(num_frames):
        # A simple heuristic to show oscillating behavior
        position, velocity = observation
        
        # Push in the direction of velocity to build momentum
        action = 0 if velocity < 0 else 2  # 0: left, 2: right
        
        # Every 20 steps, do the opposite to demonstrate oscillation
        if i % 20 == 0:
            action = 2 if action == 0 else 0
        
        # Take action and capture the rendered frame
        observation, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(Image.fromarray(frame))
        
        if terminated or truncated:
            observation, _ = env.reset(seed=42)
    
    # Save animation as GIF
    frames[0].save(
        os.path.join(output_dir, filename),
        save_all=True,
        append_images=frames[1:],
        duration=1000/fps,
        loop=0
    )
    
    env.close()
    print(f"Mountain Car GIF saved to {os.path.join(output_dir, filename)}")

if __name__ == "__main__":
    print("Generating CartPole animation...")
    create_cartpole_gif()
    
    print("Generating Mountain Car animation...")
    create_mountain_car_gif()
    
    print("GIF generation complete!") 