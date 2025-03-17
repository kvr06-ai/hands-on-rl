import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'images')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_reality_gap_diagram():
    """
    Create a minimalist diagram illustrating the sources of the reality gap
    between simulation and real-world environments.
    """
    print("Creating reality gap diagram...")
    
    # Substantially increase figure size for much better visibility
    plt.figure(figsize=(20, 12), facecolor='white')
    
    # Create a 2x2 grid with tighter spacing
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.05, hspace=0.1)
    
    # ---- Simulation world (top left) ----
    ax_sim = plt.subplot(gs[0, 0])
    ax_sim.set_xlim(0, 10)
    ax_sim.set_ylim(0, 10)
    
    # Draw a simplified robot arm in simulation - with larger components
    # Base
    ax_sim.add_patch(Rectangle((2, 1), 2, 1, facecolor='#B0B0B0', edgecolor='k', linewidth=2.5))
    # Arm segments
    ax_sim.add_patch(Rectangle((3, 2), 0.5, 3, facecolor='#4A86E8', edgecolor='k', linewidth=2.5))
    ax_sim.add_patch(Rectangle((2.5, 5), 3, 0.5, facecolor='#4A86E8', edgecolor='k', linewidth=2.5))
    # Gripper
    ax_sim.add_patch(Rectangle((5.5, 5), 0.5, 1.5, facecolor='#4A86E8', edgecolor='k', linewidth=2.5))
    ax_sim.add_patch(Rectangle((4.5, 6.5), 2.5, 0.5, facecolor='#4A86E8', edgecolor='k', linewidth=2.5))
    
    # Environment object (cube)
    ax_sim.add_patch(Rectangle((7, 3), 1.5, 1.5, facecolor='#FFD966', edgecolor='k', linewidth=2.5))
    
    # Ground
    ax_sim.add_patch(Rectangle((0, 0), 10, 1, facecolor='#EEEEEE', edgecolor='k', linewidth=2.5))
    
    # Title - much larger
    ax_sim.set_title('Simulation', fontsize=24, pad=15)
    ax_sim.axis('off')
    
    # ---- Real world (top right) ----
    ax_real = plt.subplot(gs[0, 1])
    ax_real.set_xlim(0, 10)
    ax_real.set_ylim(0, 10)
    
    # Draw a real robot arm with more details and imperfections - with larger components
    # Base with texture
    ax_real.add_patch(Rectangle((2, 1), 2, 1, facecolor='#A0A0A0', edgecolor='k', linewidth=2.5))
    # Add some texture/reflection to base - larger and more visible
    for i in range(4):
        ax_real.add_patch(Rectangle((2.2 + i*0.4, 1.2), 0.25, 0.6, facecolor='#909090', edgecolor=None, alpha=0.6))
    
    # Arm segments with imperfections
    ax_real.add_patch(Rectangle((3, 2), 0.5, 3, facecolor='#3A76D8', edgecolor='k', linewidth=2.5))
    # Add some shadows/highlights - more visible
    ax_real.add_patch(Rectangle((3.1, 2.1), 0.3, 2.8, facecolor='#2A66C8', edgecolor=None, alpha=0.4))
    ax_real.add_patch(Rectangle((2.5, 5), 3, 0.5, facecolor='#3A76D8', edgecolor='k', linewidth=2.5))
    ax_real.add_patch(Rectangle((2.6, 5.1), 2.8, 0.3, facecolor='#2A66C8', edgecolor=None, alpha=0.4))
    
    # Gripper
    ax_real.add_patch(Rectangle((5.5, 5), 0.5, 1.5, facecolor='#3A76D8', edgecolor='k', linewidth=2.5))
    ax_real.add_patch(Rectangle((4.5, 6.5), 2.5, 0.5, facecolor='#3A76D8', edgecolor='k', linewidth=2.5))
    
    # Environment object (cube)
    ax_real.add_patch(Rectangle((7, 3), 1.5, 1.5, facecolor='#EEC955', edgecolor='k', linewidth=2.5))
    # Add some texture to the cube - more prominent
    for i in range(3):
        ax_real.add_patch(Rectangle((7.1 + i*0.5, 3.1), 0.45, 0.45, facecolor='#DDC045', edgecolor=None, alpha=0.6))
    
    # Ground with texture
    ax_real.add_patch(Rectangle((0, 0), 10, 1, facecolor='#DDDDDD', edgecolor='k', linewidth=2.5))
    # Add some texture to ground - more visible
    for i in range(10):
        ax_real.add_patch(Rectangle((i, 0.1), 0.8, 0.8, facecolor='#CCCCCC', edgecolor=None, alpha=0.4))
    
    # Lighting effects (more noticeable shadow)
    shadow_polygon = Polygon([(2.5, 0.9), (5, 0.9), (5.5, 0.2), (3, 0.2)], 
                            facecolor='black', alpha=0.3)
    ax_real.add_patch(shadow_polygon)
    
    # Title - much larger
    ax_real.set_title('Reality', fontsize=24, pad=15)
    ax_real.axis('off')
    
    # ---- Gap sources (bottom) ----
    ax_gap = plt.subplot(gs[1, :])
    ax_gap.set_xlim(0, 10)
    ax_gap.set_ylim(0, 4)
    
    # Draw the three main sources of reality gap
    categories = ['Physics\nDiscrepancies', 'Sensor\nLimitations', 'Visual\nDifferences']
    x_positions = [1.7, 5, 8.3]
    
    # Create icons for each category - much larger
    for i, (category, x_pos) in enumerate(zip(categories, x_positions)):
        # Background circle - larger
        circle = Circle((x_pos, 2), 1.7, facecolor='#F3F3F3', edgecolor='#AAAAAA', alpha=0.6, linewidth=2.5)
        ax_gap.add_patch(circle)
        
        # Category name - much larger font
        ax_gap.text(x_pos, 0.4, category, ha='center', va='center', fontsize=18, color='#333333', fontweight='bold')
        
        # Icons for each category - significantly larger
        if i == 0:  # Physics
            # Stylized physics icon - force arrows and mass - larger
            mass_circle = Circle((x_pos, 2), 1.0, facecolor='#4285F4', alpha=0.8)
            ax_gap.add_patch(mass_circle)
            
            # Force arrows in different directions - much thicker and more visible
            arrow_props = dict(arrowstyle='->', linewidth=4, color='#EA4335')
            arrow1 = FancyArrowPatch((x_pos, 2), (x_pos + 1.3, 2), **arrow_props)
            arrow2 = FancyArrowPatch((x_pos, 2), (x_pos - 1.1, 2 + 1.1), **arrow_props)
            arrow3 = FancyArrowPatch((x_pos, 2), (x_pos + 0.9, 2 - 1.2), **arrow_props)
            
            ax_gap.add_patch(arrow1)
            ax_gap.add_patch(arrow2)
            ax_gap.add_patch(arrow3)
            
        elif i == 1:  # Sensors
            # Camera/sensor icon - larger
            sensor_body = Rectangle((x_pos - 0.7, 1.6), 1.4, 1.1, facecolor='#FBBC05', edgecolor='k', linewidth=2.5)
            lens = Circle((x_pos, 2.1), 0.6, facecolor='#4285F4', edgecolor='k', linewidth=2.5)
            lens_center = Circle((x_pos, 2.1), 0.25, facecolor='#34A853', edgecolor='k', linewidth=2.5)
            
            # Signal waves - much thicker and more visible
            for j, radius in enumerate(np.linspace(0.8, 1.5, 3)):
                signal = Circle((x_pos, 2.1), radius, fill=False, edgecolor='#EA4335', 
                                linestyle='--', linewidth=2.5, alpha=0.9-j*0.2)
                ax_gap.add_patch(signal)
                
            ax_gap.add_patch(sensor_body)
            ax_gap.add_patch(lens)
            ax_gap.add_patch(lens_center)
            
        else:  # Visual
            # Simple representation of different appearances - much larger elements
            # Sim view (simplified)
            sim_view = Rectangle((x_pos - 0.9, 1.4), 0.8, 1.2, facecolor='#FBBC05', edgecolor='k', linewidth=2.5)
            ax_gap.add_patch(sim_view)
            
            # Real view (with texture)
            real_view = Rectangle((x_pos + 0.1, 1.4), 0.8, 1.2, facecolor='#EA4335', edgecolor='k', linewidth=2.5)
            ax_gap.add_patch(real_view)
            
            # Texture lines on real view - more visible and prominent
            for j in range(3):
                line = Rectangle((x_pos + 0.15, 1.55 + j*0.35), 0.7, 0.12, facecolor='#AA2315')
                ax_gap.add_patch(line)
            
            # Visual connection - much thicker lines
            ax_gap.plot([x_pos - 0.55, x_pos + 0.55], [3.1, 3.1], '-', color='#4285F4', linewidth=3.5)
            ax_gap.plot([x_pos - 0.55, x_pos - 0.55], [2.6, 3.1], '-', color='#4285F4', linewidth=3.5)
            ax_gap.plot([x_pos + 0.55, x_pos + 0.55], [2.6, 3.1], '-', color='#4285F4', linewidth=3.5)
            
    # Title for the bottom section - much larger font
    ax_gap.text(5, 3.7, 'Sources of the Reality Gap', ha='center', va='center', fontsize=24, fontweight='bold')
    ax_gap.axis('off')
    
    # Arrows connecting the top diagrams to the gap visualization - much thicker and more prominent
    arrow_sim_to_gap = FancyArrowPatch((3, 0.5), (3, 3.5), 
                                      connectionstyle="arc3,rad=-0.2",
                                      arrowstyle='->', linewidth=3.5, color='#4285F4')
    arrow_real_to_gap = FancyArrowPatch((7, 0.5), (7, 3.5), 
                                       connectionstyle="arc3,rad=0.2",
                                       arrowstyle='->', linewidth=3.5, color='#EA4335')
    
    plt.gcf().add_artist(arrow_sim_to_gap)
    plt.gcf().add_artist(arrow_real_to_gap)
    
    # Save the figure with higher resolution and tight bounding box
    plt.tight_layout(pad=0.5)
    output_path = os.path.join(output_dir, 'reality_gap_diagram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Reality gap diagram saved to {output_path}")


def create_domain_randomization_visualization():
    """
    Create a visualization showing domain randomization across multiple
    simulation instances to create policies robust to reality.
    """
    print("Creating domain randomization visualization...")
    
    # Substantially increase figure size for much better visibility
    plt.figure(figsize=(20, 12), facecolor='white')
    
    # Setup a 3x3 grid + space for the real world at the bottom - with tighter spacing
    gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 1.2], hspace=0.15, wspace=0.05)
    
    # Generate variations for different simulator parameters
    param_variations = {
        'lighting': ['dim', 'normal', 'bright', 'warm', 'cool'],
        'texture': ['simple', 'detailed', 'noisy', 'smooth'],
        'object_position_x': [3, 3.5, 2.5, 4, 2],
        'object_position_y': [3, 3.5, 2.5, 4, 2],
        'camera_angle': [0, 5, -5, 10, -10],
        'friction': [0.3, 0.5, 0.7, 0.9, 1.1],
        'gravity': [9.5, 9.8, 10.1, 10.4]
    }
    
    # Colors for different simulation variations
    sim_colors = ['#E6F2FF', '#F0F9E8', '#FFF2E6', '#F9E8F0', '#E8F0F9', '#F2FFE6', '#FFE6F2', '#E6FFE8', '#F2E6FF']
    
    # Create 9 randomized simulations
    np.random.seed(42)  # For reproducibility
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            ax = plt.subplot(gs[i, j])
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 6)
            
            # Set sim-specific background color
            ax.set_facecolor(sim_colors[idx])
            
            # Randomly select variations for this simulation
            lighting = np.random.choice(param_variations['lighting'])
            texture = np.random.choice(param_variations['texture'])
            obj_pos_x = np.random.choice(param_variations['object_position_x'])
            obj_pos_y = np.random.choice(param_variations['object_position_y'])
            camera_angle = np.random.choice(param_variations['camera_angle'])
            friction = np.random.choice(param_variations['friction'])
            
            # Background environment - floor - thicker line
            floor = Rectangle((0, 0), 10, 1, facecolor='#DDDDDD', alpha=0.8, linewidth=2.5)
            ax.add_patch(floor)
            
            # Add environment textures based on randomization - more visible
            if texture == 'detailed' or texture == 'noisy':
                for k in range(20):
                    # Random texture elements on floor
                    x, y = np.random.uniform(0, 9.5), np.random.uniform(0, 0.9)
                    size = np.random.uniform(0.15, 0.45)
                    alpha = np.random.uniform(0.15, 0.35)
                    rect = Rectangle((x, y), size, size, facecolor='#AAAAAA', alpha=alpha)
                    ax.add_patch(rect)
            
            # Draw a robot based on randomization - simple arm for illustration - larger components
            # Base
            base = Rectangle((2, 1), 1.5, 0.8, facecolor='#B0B0B0', edgecolor='k', alpha=0.9, linewidth=2.5)
            # Arm
            arm_1 = Rectangle((2.5, 1.8), 0.5, 2, facecolor='#4A86E8', edgecolor='k', alpha=0.9, linewidth=2.5)
            arm_2 = Rectangle((2, 3.8), 1.5, 0.5, facecolor='#4A86E8', edgecolor='k', alpha=0.9, linewidth=2.5)
            # End effector
            end_effector = Circle((2, 4.3), 0.5, facecolor='#EA4335', edgecolor='k', alpha=0.9, linewidth=2.5)
            
            # Apply camera angle variations
            if camera_angle != 0:
                # Modify positions based on perspective/angle
                angle_factor = camera_angle / 10  # Scale down the effect
                arm_1 = Rectangle((2.5 + angle_factor*0.2, 1.8), 
                                 0.5 - abs(angle_factor)*0.1, 2, 
                                 facecolor='#4A86E8', edgecolor='k', alpha=0.9, linewidth=2.5)
            
            ax.add_patch(base)
            ax.add_patch(arm_1)
            ax.add_patch(arm_2)
            ax.add_patch(end_effector)
            
            # Target object with randomized position - larger
            target = Rectangle((obj_pos_x + 5, obj_pos_y), 1, 1, facecolor='#FFD966', edgecolor='k', alpha=0.9, linewidth=2.5)
            ax.add_patch(target)
            
            # Lighting effects - more pronounced
            if lighting == 'dim':
                # Add dark overlay
                ax.add_patch(Rectangle((0, 0), 10, 6, facecolor='black', alpha=0.25))
            elif lighting == 'bright':
                # Add highlight
                light_circle = Circle((5, 5), 3, facecolor='yellow', alpha=0.15)
                ax.add_patch(light_circle)
            elif lighting == 'warm':
                # Add warm overlay
                ax.add_patch(Rectangle((0, 0), 10, 6, facecolor='#FF9900', alpha=0.15))
            elif lighting == 'cool':
                # Add cool overlay
                ax.add_patch(Rectangle((0, 0), 10, 6, facecolor='#3D85C6', alpha=0.15))
                
            # Add a text label showing key parameters that were randomized - much larger font
            param_text = f"L:{lighting[:1].upper()}, T:{texture[:1].upper()}, F:{friction:.1f}"
            ax.text(5, 5.5, param_text, ha='center', fontsize=14, 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Turn off axes
            ax.axis('off')
    
    # ---- Add real world at the bottom ----
    ax_real = plt.subplot(gs[3, :])
    ax_real.set_xlim(0, 10)
    ax_real.set_ylim(0, 4)
    
    # Create more detailed real-world scene
    # Floor with texture - thicker line
    floor = Rectangle((0, 0), 10, 1, facecolor='#CCCCCC', edgecolor='k', linewidth=2.5)
    ax_real.add_patch(floor)
    
    # Add realistic floor texture - more visible
    for i in range(20):
        for j in range(4):
            x, y = i*0.5, j*0.25
            alpha = np.random.uniform(0.08, 0.2)
            rect = Rectangle((x, y), 0.48, 0.23, facecolor='#AAAAAA', alpha=alpha)
            ax_real.add_patch(rect)
    
    # Real robot with more details - larger components
    # Base
    base = Rectangle((2, 1), 1.5, 0.8, facecolor='#909090', edgecolor='k', linewidth=2.5)
    # Add details to base - more visible
    for i in range(3):
        detail = Rectangle((2.2 + i*0.4, 1.1), 0.3, 0.6, facecolor='#808080', edgecolor=None, alpha=0.6)
        ax_real.add_patch(detail)
    
    # Arm segments - thicker lines
    arm_1 = Rectangle((2.5, 1.8), 0.5, 2, facecolor='#3A76D8', edgecolor='k', linewidth=2.5)
    arm_2 = Rectangle((2, 3.8), 1.5, 0.5, facecolor='#3A76D8', edgecolor='k', linewidth=2.5)
    
    # Add details/shading to arms - more visible
    ax_real.add_patch(Rectangle((2.6, 1.9), 0.3, 1.8, facecolor='#2A66C8', edgecolor=None, alpha=0.4))
    ax_real.add_patch(Rectangle((2.1, 3.9), 1.3, 0.3, facecolor='#2A66C8', edgecolor=None, alpha=0.4))
    
    # End effector - thicker lines
    end_effector = Circle((2, 4.3), 0.5, facecolor='#D03223', edgecolor='k', linewidth=2.5)
    # Add details to end effector - more visible
    end_detail = Circle((2, 4.3), 0.3, facecolor='#E04233', edgecolor=None, alpha=0.6)
    
    # Target object - thicker lines
    target = Rectangle((7, 3), 1, 1, facecolor='#EEC955', edgecolor='k', linewidth=2.5)
    # Add texture to target - more visible
    for i in range(2):
        for j in range(2):
            detail = Rectangle((7.1 + i*0.45, 3.1 + j*0.45), 0.4, 0.4, facecolor='#DDC045', edgecolor=None, alpha=0.4)
            ax_real.add_patch(detail)
    
    # Add robot components to the scene
    ax_real.add_patch(base)
    ax_real.add_patch(arm_1)
    ax_real.add_patch(arm_2)
    ax_real.add_patch(end_effector)
    ax_real.add_patch(end_detail)
    ax_real.add_patch(target)
    
    # Add shadow under robot - more visible
    shadow = Polygon([(1.5, 0.9), (3.5, 0.9), (4, 0.1), (1, 0.1)], facecolor='black', alpha=0.3)
    ax_real.add_patch(shadow)
    
    # Turn off axes for real world
    ax_real.axis('off')
    
    # Add big arrow from simulations to real world - much more prominent
    arrow_props = dict(arrowstyle='->', linewidth=5, color='#4285F4', connectionstyle='arc3,rad=0.0')
    big_arrow = FancyArrowPatch((5, 2), (5, 3), **arrow_props, mutation_scale=50)
    plt.gcf().add_artist(big_arrow)
    
    # Add titles and explanations - much larger fonts
    plt.figtext(0.5, 0.95, 'Domain Randomization', fontsize=26, ha='center', fontweight='bold')
    plt.figtext(0.5, 0.91, 'Training across varied simulations to create policies robust to reality', 
               fontsize=18, ha='center', fontstyle='italic')
    
    plt.figtext(0.5, 0.03, 'Real World', fontsize=24, ha='center', fontweight='bold')
    plt.figtext(0.5, 0.005, 'Policy transfers successfully despite never seeing real data', 
               fontsize=16, ha='center', fontstyle='italic')
    
    # Save the figure with higher resolution and tight bounding box
    plt.tight_layout(rect=[0, 0.05, 1, 0.9], pad=0.5)
    output_path = os.path.join(output_dir, 'domain_randomization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Domain randomization visualization saved to {output_path}")


if __name__ == "__main__":
    create_reality_gap_diagram()
    create_domain_randomization_visualization()
    print("Simulation-to-Real Transfer visualizations complete!") 