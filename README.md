# Hands-on Reinforcement Learning

A comprehensive, interactive guide to reinforcement learning environments, from simple benchmarks like CartPole to complex physics simulations with MuJoCo and beyond. You can read the entire article on my website: https://kaushikrajan.me/blog/reinforcement-learning/

## Project Overview

This repository contains an in-depth, web-based article exploring various reinforcement learning environments. It provides detailed technical explanations, visualizations, and code examples for various RL environments - from classic control problems to cutting-edge GPU-accelerated simulators.

The project is structured as a self-contained web application with:
- Interactive HTML-based content with 12 sections covering different aspects of RL environments
- Visualizations created specifically for each concept using Python and Matplotlib
- Interactive elements that showcase RL algorithms and environments
- Code examples that readers can adapt for their own projects

## Article Structure

The article is organized into 12 comprehensive sections:

1. **Introduction: The Simulation Frontier of AI** - Overview of RL and simulation environments
2. **Reinforcement Learning Fundamentals: A Technical Primer** - Mathematical foundations of RL
3. **The Classic Benchmarks: CartPole and Beyond** - Analysis of foundational control problems
4. **Atari Learning Environment: The Deep RL Revolution** - Pixel-based RL and DQN breakthroughs
5. **Continuous Control: Entering the Robotics Domain** - Challenges of continuous action spaces
6. **MuJoCo Physics: Advanced Simulation for Robotics** - Detailed coverage of physics-based simulations
7. **Multi-Agent Environments: From Competition to Cooperation** - Exploring multi-agent interactions
8. **Procedural Generation and Generalization** - Techniques for creating varying environments
9. **Simulation-to-Real Transfer: Bridging the Reality Gap** - Methods for transferring learned policies
10. **The Cutting Edge: Modern Simulation Platforms** - Survey of recent simulation technologies
11. **Practical Implementation: Building Your Own RL Pipeline** - Hands-on engineering best practices
12. **Conclusion: The Future of RL Environments** - Emerging trends and future directions

## Visualizations and Code Examples

The project includes over 20 custom visualization scripts in the `code/examples` directory that generate:

- **Static diagrams**: MDP representations, algorithm taxonomies, reward shaping comparisons
- **Animated GIFs**: Classic control problems, Atari games, multi-agent scenarios
- **Interactive elements**: Environment comparisons, algorithm benchmarks
- **Statistical analyses**: Learning curves, training statistics, performance metrics

Each script is self-contained and can be run independently to generate visualizations for specific RL concepts.

### Key Visualization Scripts

- `create_rl_cycle_gif.py` - Animated illustration of the RL loop (agent-environment interaction)
- `create_classic_control_gifs.py` - Animations of CartPole and MountainCar environments
- `create_atari_breakout_gif.py` - Visualization of Atari Breakout gameplay
- `create_mujoco_visualizations.py` - Detailed illustrations of MuJoCo physics simulations
- `create_multiagent_visualizations.py` - Diagrams of multi-agent interactions
- `create_sim_to_real_visualizations.py` - Visual explanations of reality gap concepts
- `create_modern_sim_visualizations.py` - Visualizations of cutting-edge simulation platforms

## File Structure

```
hands-on-rl/
├── README.md                     # This documentation file
├── index.html                    # Main HTML file that loads article sections
├── serve.py                      # Simple HTTP server for local viewing
├── requirements.txt              # Python package dependencies
├── .gitignore                    # Specifies untracked files
├── code/                         # Python code directory
│   └── examples/                 # Visualization and example scripts
│       ├── create_rl_cycle_gif.py
│       ├── create_mdp_diagram.py
│       ├── create_rl_taxonomy.py
│       └── ... (20+ visualization scripts)
├── article_sections/             # HTML content for each article section
│   ├── article_section_1.html
│   ├── article_section_2.html
│   ├── section_3.html
│   └── ... (sections 1-12)
└── images/                       # Generated visualization images
    ├── rl_cycle.gif
    ├── mdp_diagram.png
    └── ... (various visualizations)
```

## .gitignore and Untracked Files

The repository uses a `.gitignore` file that excludes certain directories to keep the repository size manageable while still allowing users to regenerate content locally. The following are **not** tracked in version control:

```
/article_sections/    # HTML content for each section
/images/              # Generated visualization images
```

**Important Note for Users:** When cloning this repository, you will need to:

1. Run the visualization scripts to generate the images locally
2. Serve the application using the provided `serve.py` script

The design choice to exclude these directories from version control was made to:
- Keep the repository size manageable (generated images can be large)
- Encourage users to run the code examples themselves
- Avoid potential licensing issues with generated content

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hands-on-rl.git
cd hands-on-rl
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Generate Visualizations

Run the visualization scripts to generate the images needed for the article:

```bash
# General approach for running all scripts
for script in code/examples/*.py; do python "$script"; done

# Or run individual scripts as needed
python code/examples/create_rl_cycle_gif.py
python code/examples/create_mdp_diagram.py
# etc.
```

## Running the Article Locally

Start the local server to view the article:

```bash
python serve.py
```

Then open your browser and navigate to [http://localhost:8001](http://localhost:8001)

## Additional Features

- **Interactive Elements**: The article includes interactive elements like hoverable tables and dynamic comparisons
- **Comprehensive Code Comments**: All scripts are thoroughly documented with detailed comments
- **Educational Progression**: The article is structured to build knowledge incrementally
- **Research References**: Includes citations to key papers in reinforcement learning

## Future Additions

Planned future enhancements to this project include:
- Additional interactive demonstrations of RL algorithms
- Downloadable Jupyter notebooks for each section
- Integration with online RL environments
- Expanded coverage of newer simulation platforms

## License

This project is released under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project draws inspiration from many open-source RL frameworks, including:
- OpenAI Gym/Gymnasium
- DeepMind Control Suite
- MuJoCo Physics Engine
- PettingZoo Multi-Agent Framework

## Citing This Work

If you use this content in your research or teaching, please cite it as:

```
@misc{handsonrl2025,
  author = {Kaushik Rajan},
  title = {Hands-on Reinforcement Learning: From CartPole to MuJoCo},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/kvr06-ai/hands-on-rl}
}
``` 
