# Implementation and Analysis of PPO Algorithm for a Lunar Landing RL Game

## Authors
- **Adam Gear**
- **Maksat Omorov**
- **Bernar Yerken**

## Abstract
This study focuses on implementing and analyzing the **Proximal Policy Optimization-Clip (PPO-Clip)** algorithm in the Lunar Lander environment using the **Gymnasium** API. Developed for the GE2340 AI: Past, Present and Future course, the project addresses three primary objectives:
1. **Understanding the Mathematical Foundation of PPO-Clip:**
   - Exploring the theoretical underpinnings of the PPO-Clip algorithm, including policy gradients, clipping mechanisms, and the optimization process.
2. **Demonstrating Practical Implementation in the Lunar Lander Scenario:**
   - Implementing the PPO-Clip algorithm within the LunarLander environment using **Stable Baselines3**, ensuring a functional RL agent capable of interacting with the environment.
3. **Optimizing Performance through Hyperparameter Tuning:**
   - Systematically adjusting key hyperparameters—**timesteps**, **learning rate**, and **network architecture**—to evaluate their impact on the agent's performance and identify optimal configurations.

The study examines the influence of key parameters, such as learning rate and evaluation timesteps, on the algorithm's efficiency and outcomes. Utilizing frameworks like **Stable Baselines3**, **TensorBoard**, and **Pandas**, the research evaluates the impact of these parameters on the agent's ability to successfully land the lunar module. The findings highlight optimal configurations that enhance training efficiency and landing success rates, providing valuable insights into the effectiveness of PPO-Clip in complex RL tasks.

## Table of Contents
- [Project Title](#implementation-and-analysis-of-ppo-algorithm-for-a-lunar-landing-rl-game)
- [Authors](#authors)
- [Abstract](#abstract)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Motivation](#motivation)
- [Related Work](#related-work)
- [Experiments](#experiments)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusions and Limitations](#conclusions-and-limitations)
- [Installation Instructions](#installation-instructions)
  - [Dependencies](#dependencies)
  - [Installation Steps](#installation-steps)
- [Usage](#usage)
  - [Execution Commands](#execution-commands)
  - [Visualization](#visualization)
- [Configuration](#configuration)
  - [Default Settings](#default-settings)
  - [Timesteps experiment](#timesteps-experiment)
  - [Learning rate experiment](#learning-rate-experiment)
  - [Architecture experiment](#architecture-experiment)
- [Acknowledgements](#acknowledgements)
- [FAQ](#faq)
- [Contact Information](#contact-information)

## Introduction
The **LunarLander** environment, provided by the **Gymnasium** API, simulates a rocket landing on uneven terrain, with rewards and penalties based on landing success and efficiency. It serves as an excellent test bed for evaluating reinforcement learning (RL) algorithms due to its combination of continuous state and action spaces, stochastic dynamics, and the necessity for precise control.

### Reward System
- **+100 to +140 points** for landing within the target zone.
- **-100 points** for crashing.
- **+100 points** for a safe out-of-bounds landing.
- **-0.3 points per frame** for using rockets to slow the ascent.

This straightforward reward structure allows for the evaluation of RL algorithm performance by balancing successful landings with efficient resource usage.

## Objectives
The primary objectives of this project are:

1. **Understanding the Mathematical Foundation of PPO-Clip:**
   - Explore the theoretical underpinnings of the PPO-Clip algorithm, including policy gradients, clipping mechanisms, and the optimization process.

2. **Demonstrating Practical Implementation in the Lunar Lander Scenario:**
   - Implement the PPO-Clip algorithm within the LunarLander environment using **Stable Baselines3**, ensuring a functional RL agent capable of interacting with the environment.

3. **Optimizing Performance through Hyperparameter Tuning:**
   - Systematically adjust key hyperparameters—**timesteps**, **learning rate**, and **network architecture**—to evaluate their impact on the agent's performance and identify optimal configurations.

## Motivation
The **PPO-Clip algorithm** was chosen for its proven effectiveness in handling complex RL tasks with stability and reliability. PPO-Clip strikes a balance between exploration and exploitation by preventing significant policy updates, thereby avoiding performance degradation. The **LunarLander** environment offers a robust platform to test and fine-tune PPO-Clip, providing insights into its adaptability and robustness in handling continuous control tasks with high-dimensional state spaces. Understanding how hyperparameter adjustments influence performance further contributes to the development of more efficient and effective RL agents.

## Related Work
Reinforcement learning differs from supervised learning in that RL training data evolves dynamically as the agent's policy changes. This variability can cause instability, especially with large learning rates.

### PPO-Clip Algorithm
- **Developed by OpenAI in 2017** for Atari games and later adapted for applications such as ChatGPT.
- **Key Components:**
  - **Advantage Function (\( A_t \ )):** Measures the difference between actual rewards and expected rewards, guiding learning.
  - **Probability Ratio (\( r(\theta) \ )):** Evaluates the likelihood of an action under the current and previous policies, ensuring stability.

### Objective Function
The PPO-Clip objective function ensures updates do not deviate significantly from previous policies:
\[
E_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right]
\]

### Final Loss Function
Combines the objective function with terms for value function approximation and entropy regularization.

### Advantages
- **Simplicity and Compactness:** Compared to Trust Region Policy Optimization (TRPO).
- **Tunability:** Allows for performance optimization through hyperparameter adjustments.
- **Strong Performance:** Demonstrated efficacy in various tasks.

### Drawbacks
- **Slower Learning:** Prioritizes stability over speed.
- **Limited Sample Efficiency:** Compared to off-policy algorithms like Q-learning.

Studies such as [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) have demonstrated PPO-Clip's superiority over other policy gradient methods in various environments. In the context of the LunarLander environment, previous works have utilized PPO and other algorithms like Deep Q-Networks (DQN) to achieve varying degrees of success. However, comprehensive analyses focusing on hyperparameter optimization specifically for PPO-Clip within this environment are limited. This project builds upon existing research by providing an in-depth exploration of how key hyperparameters influence the learning dynamics and performance of PPO-Clip in the LunarLander task.

## Experiments
Four experiments were conducted to analyze the PPO-Clip algorithm by tuning its hyperparameters:

1. **Baseline (PPO Default):**
   - Served as the reference model for comparison.
   - Utilized the default hyperparameters provided by **Stable Baselines3** for PPO-Clip.

2. **Training Time Steps:**
   - Models were tested with half and double the default training steps (500k and 2 million timesteps) to observe performance differences.

3. **Learning Rate:**
   - The learning rate was increased and decreased by a factor of 10 (×10 and ×0.1) to assess its effect on performance.

4. **Architectural Modifications:**
   - Adjustments to the number of neurons, layers, and activation functions (from tanh to ReLU) were explored.
   - Variations included different layer configurations such as 2-layered 128/64, 2-layered 64/128, 2-layered, and 3-layered architectures. ** More details about the layers and configurations in the main.ipynb file **

## Evaluation
### A. Overview of Results
The experiments evaluated performance using mean rewards and standard deviations.

| **Model**                  | **Mean Reward** | **Standard Deviation** |
|----------------------------|-----------------|------------------------|
| PPO Default                | 261.04          | 41.43                  |
| PPO 500k timesteps         | 226.77          | 64.12                  |
| PPO 2mil timesteps         | 267.35          | 22.86                  |
| PPO lr ×10                 | 235.93          | 83.44                  |
| PPO lr ×0.1                | 271.72          | 24.65                  |
| PPO 2-layered 128/64       | 257.01          | 25.95                  |
| PPO 2-layered 64/128       | 244.97          | 46.34                  |
| PPO 2-layered              | 275.69          | 26.83                  |
| PPO 3-layered              | 279.25          | 19.96                  |
| PPO ReLU                   | 279.57          | 19.76                  |
| PPO possibly best          | 270.82          | 43.24                  |
| PPO 2nd possibly best      | 280.86          | 37.39                  |

### B. Timesteps Experiment
- **Doubling the Training Steps:**  
  Increasing the number of timesteps from 500k to 2 million led to an improvement in mean rewards from 226.77 to 267.35 and a reduction in standard deviation from 64.12 to 22.86. This indicates enhanced learning stability and performance with extended training periods. However, diminishing returns were observed beyond a certain point.

### C. Learning Rate Experiment
- **Higher Learning Rates (×10):**  
  - Achieved a mean reward of 235.93 with a high standard deviation of 83.44.
  - Indicates instability due to over-correction in policy updates.
  
- **Lower Learning Rates (×0.1):**  
  - Achieved a mean reward of 271.72 with a standard deviation of 24.65.
  - Resulted in more stable and consistent performance, enhancing the agent's ability to converge effectively.

### D. Architectural Modifications
- **Layer and Neuron Adjustments:**  
  - **2-layered 128/64:** Mean Reward: 257.01, Std Dev: 25.95
  - **2-layered 64/128:** Mean Reward: 244.97, Std Dev: 46.34
  - **2-layered:** Mean Reward: 275.69, Std Dev: 26.83
  - **3-layered:** Mean Reward: 279.25, Std Dev: 19.96
  - **ReLU Activation:** Mean Reward: 279.57, Std Dev: 19.76

- **Findings:**  
  Adding layers and neurons improved performance, with the 3-layered and ReLU activation models outperforming others. ReLU activation outperformed the default tanh function, producing higher rewards and lower standard deviations.

### E. Best Model
- **Combination of Optimized Parameters:**  
  The "best models" combined ReLU activation, three layers with 128 neurons each, and 2 million training steps. The first one has 0.1x learning rate, while the second has the default.
  
- **Performance:**  
  - Achieved high mean rewards (279.25 to 280.86)
  - Standard deviations indicate occasional extreme scores but overall robust performance.

### F. Landing Time Analysis
- **Early Training:**  
  Models crashed frequently, resulting in negative rewards.
  
- **Mid Training:**  
  Models became overly cautious, increasing landing times to avoid crashes.
  
- **Post Training:**  
  Policies were refined to reduce landing times while maintaining safety, leading to more efficient landings.

## Results
Our experiments yielded the following key findings:

- **Baseline Performance:**
  - The default PPO-Clip settings achieved an average reward of **261.04** with a standard deviation of **41.43**, serving as a reference point for subsequent experiments.

- **Training Time Steps:**
  - **500k Timesteps:**  
    - **Mean Reward:** 226.77  
    - **Standard Deviation:** 64.12  
  - **2 Million Timesteps:**  
    - **Mean Reward:** 267.35  
    - **Standard Deviation:** 22.86  
  - **Insight:** Increasing training timesteps improved mean rewards and reduced variability, enhancing stability and performance.

- **Learning Rate:**
  - **Learning Rate ×10:**  
    - **Mean Reward:** 235.93  
    - **Standard Deviation:** 83.44  
    - **Insight:** Higher learning rates led to greater reward variance, indicating instability.
  - **Learning Rate ×0.1:**  
    - **Mean Reward:** 271.72  
    - **Standard Deviation:** 24.65  
    - **Insight:** Lower learning rates resulted in more stable and consistent performance.

- **Architectural Modifications:**
  - **2-layered 128/64:**  
    - **Mean Reward:** 257.01  
    - **Standard Deviation:** 25.95  
  - **2-layered 64/128:**  
    - **Mean Reward:** 244.97  
    - **Standard Deviation:** 46.34  
  - **2-layered:**  
    - **Mean Reward:** 275.69  
    - **Standard Deviation:** 26.83  
  - **3-layered:**  
    - **Mean Reward:** 279.25  
    - **Standard Deviation:** 19.96  
  - **ReLU Activation:**  
    - **Mean Reward:** 279.57  
    - **Standard Deviation:** 19.76  
  - **Insight:** Adding layers and utilizing ReLU activation significantly improved performance, leading to higher rewards and lower variability. Policy layers are better than value function layers, since increasing the amount of neurons in the value function resulted in inconsistent landings, while increased amount of neurons in policy function worked just right.

- **Best Models:**
  - **PPO possibly best:**  
    - **Mean Reward:** 270.82  
    - **Standard Deviation:** 43.24  
  - **PPO 2nd possibly best:**  
    - **Mean Reward:** 280.86  
    - **Standard Deviation:** 37.39  
  - **Insight:** Combining ReLU activation, three layers with 128 neurons each, and 2 million training steps yielded the highest mean rewards, demonstrating the effectiveness of these hyperparameter configurations. The difference between the models is the learning rate (0.1x in the first, default in the second), since decreased learning rate affected stability.

- **Landing Time Analysis:**
  - Models initially experienced frequent crashes.
  - Over time, models became cautious, leading to increased landing times.
  - With refined policies, landing times decreased, achieving more efficient and successful landings.


## Conclusions and Limitations

### Conclusions
- **Training Timesteps:** Longer training steps improved performance and stability, though with diminishing returns beyond 2 million timesteps.
- **Learning Rate:** Higher learning rates increased reward variance and instability, while lower learning rates enhanced stability and performance consistency.
- **Architectural Enhancements:** Adding layers and switching to ReLU activation significantly boosted performance, resulting in higher rewards and reduced variability.
- **Best Configurations:** The combination of ReLU activation, three layers with 128 neurons each, and 2 million training steps achieved the highest mean rewards, indicating an optimal balance between model complexity and training duration.

### Limitations
- **Hyperparameter Scope:** Not all hyperparameters, such as discounted rewards and entropy terms, were explored, potentially limiting the optimization scope.
- **Comparative Analysis:** The study did not include comparisons with alternative RL algorithms like Q-learning, which could provide broader insights into PPO-Clip's performance.
- **Framework Depth:** A deeper exploration of the Gymnasium framework and building a model from scratch could offer further understanding and optimization opportunities.
- **Computational Resources:** Limited computational resources constrained the depth and breadth of hyperparameter exploration.

## Installation Instructions

### Dependencies
Ensure you have the following frameworks and libraries installed:
- **Python 3.7+**
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Pandas](https://pandas.pydata.org/)

### Installation Steps
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/ppo-lunar-lander.git
   cd ppo-lunar-lander
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Execution Commands
- **Train the RL Agent:**
  ```python
  model = train_agent(algo=PPO, timesteps=*put_timesteps_here*, log_name='put_log_name_here_for_tensorboard',
                      policy='MlpPolicy', log_path='data_logs', lr=0.0003)
  ```

- **Evaluate the Trained Model:**
  ```python
  mean_reward, std_reward = evaluate_rl_model(algo=PPO, model_path=*path_here*, n_eval_episodes=100, deterministic=True)
  ```

### Visualization
- **Launch TensorBoard to Visualize Training Metrics:**
  ```bash
  tensorboard --logdir=data/logs
  ```

   *Open your web browser and navigate to `http://localhost:6006` to view the TensorBoard dashboard.*

- **Watch the visualization of the agent's gameplay**
  ```python
  visualize_agent(model_path=*path_here*, episodes=5, algo=PPO):
  ```


## Configuration

### Default Settings
- **Timesteps:** `1000000`
- **Learning Rate:** `0.0003`
- **Architecture:**
  - **Layers:** `[64, 64]` (unless mentioned, both value and policy have same architecture)
  - **Activation:** `tanh`

### Timesteps experiment
- **Timesteps:** `500000` or `2000000`

### Learning rate experiment
- **Learning rate:** `0.003` or `0.00003`

### Architecture experiment

A. Policy function vs Value function:
- **Architecture:** 
  - **Layers:** `Policy function = [128, 128] and ]Value function = [64, 64]` or `Policy function = [64, 64] and ]Value function = [128, 128]`

B. Increased neuron layers
- **Architecture:**
  - **Layers:** `[128, 128]` or `[128, 128, 128]` 

C. Activation function
- **Architecture:**
  - **Activation:** `ReLU`

D. Best models:
- **Timesteps:** `2000000`
- **Learning Rate:** `0.0003` or `0.00003`
- **Architecture:**
  - **Layers:** `[128, 128, 128]`
  - **Activation:** `ReLU`


## Acknowledgements
- **Course:** GE2340 AI: Past, Present and Future
- **Frameworks and Libraries:** Gymnasium, Stable Baselines3, TensorBoard, Pandas
- **Inspirations:** OpenAI's PPO research, various RL tutorials, and academic papers on reinforcement learning.

## FAQ

**Q1: How do I change the hyperparameters?**  
*A1:* Modify the train_model function with your desired settings.

**Q2: How can I visualize the training progress?**  
*A2:* Launch TensorBoard using the command `tensorboard --logdir=data/logs` and navigate to `http://localhost:6006`.

**Q3: I'm encountering installation issues. What should I do?**  
*A3:* Ensure you have Python 3.7+ installed and that all dependencies are correctly listed in `requirements.txt`.

**Q4: Can I use a different RL algorithm?**  
*A4:* Yes, you can integrate other algorithms supported by Stable Baselines3 by modifying the training functions accordingly (i.e. change the algo parameter).

**Q5: What do the standard deviations in the results indicate?**  
*A5:* The standard deviations reflect the variability in the rewards obtained across different training runs. Lower standard deviations indicate more consistent performance.

## Contact Information
For further inquiries or support, please contact the authors:

- **Adam Gear:** 
- **Maksat Omorov:** momorov2-c@my.cityu.edu.hk (responsible for the github and the coding)
- **Bernar Yerken:**


---

**Repository Structure:**
- `data/logs`: Contains logs for the model's training.
- `models`: Stores trained models.
