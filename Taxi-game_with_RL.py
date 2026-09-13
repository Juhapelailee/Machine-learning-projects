""" Juha Närhi
Reinforcement learning (Gymnasium)

This script utilizes Q-learning to solve the "Taxi-v3" environment from the OpenAI Gym library,
which simulates the movement of a taxi in a city.

Q-learning Parameters:
Parameters such as alpha, gamma, and epsilon are specified for Q-learning,
controlling the learning rate, discount factor for future rewards,
and the exploration rate using the epsilon-greedy strategy.

Training:
Q-learning training is performed 10 times (for run in range(10)).
For each training run, an empty Q-table (Q_reward) is initialized and updated based on actions taken in the environment.

Saving Q-tables:
After each training run, the Q-table is stored in the qtables_list for later evaluation.

Evaluation:
The script evaluates the trained model's performance using the saved Q-tables.
It executes 50 interactions with the environment for each Q-table,
 recording metrics such as total reward and the number of interactions.

Calculating and Printing Averages:
Finally, the script calculates and prints average performance metrics,
including the average total reward and the average number of interactions, across all Q-tables.

The purpose of this script is to train a Q-learning model in the "Taxi-v3"-
environment and assess its performance over multiple training runs, utilizing Q-tables for evaluation.

"""
import gym
import numpy as np
import time

# Environment
env = gym.make("Taxi-v3", render_mode="ansi")

# Training parameters for Q learning
alpha = 0.9  # Learning rate
gamma = 0.9  # Future reward discount factor
epsilon = 0.1
num_of_episodes = 1000
num_of_steps = 500  # per each episode

# List to store Q-tables after each training run
qtables_list = []

# Run training part 10 times
for run in range(10):
    # Q table for rewards
    Q_reward = -100000*np.zeros((500, 6))

    # Training w/ epsilon-greedy
    for episode in range(num_of_episodes):
        state = env.reset()[0]
        tot_reward = 0

        for step in range(num_of_steps):
            # Epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q_reward[state, :])  # Exploit

            # Taking the action and observing the next state and reward
            new_state, reward, done, truncated, info = env.step(action)
            tot_reward += reward

            # Updating Q value using the Q-learning update rule
            Q_reward[state, action] = Q_reward[state, action] + alpha * (
                        reward + gamma * np.max(Q_reward[new_state, :]) - Q_reward[state, action])
            # Move to the next state
            state = new_state
            if done:
                break

        # Store the Q-table in the list after each training run
    qtables_list.append(Q_reward)
    # printing the number of Q-tables so it will be easier to understand
    print(len(qtables_list))
# Evaluation
average_rewards = []
average_interactions = []

# Evaluate each Q-table
number_of_qtable = 0
for qtable in qtables_list:
    number_of_qtable += 1
    rewards = []
    num_of_interactions = 0
    state = env.reset()[0]
    tot_reward = 0
    for t in range(50):
        num_of_interactions += 1
        action = np.argmax(qtable[state, :])
        state, reward, done, truncated, info = env.step(action)
        tot_reward += reward
        print(env.render())
        time.sleep(0.001)
        if done:
            print(f"Total reward for Q-table number {number_of_qtable} is {tot_reward}")
            print("Total interactions for Q-table number", number_of_qtable, "is", num_of_interactions)
            break

    # Store results for each Q-table
    average_rewards.append(tot_reward)
    average_interactions.append(num_of_interactions)

# Calculate and print averages
print()
print("TOTAL REWARD AND NUMBER OF INTERACTIONS")
print("Average total reward for one Q-table", np.mean(average_rewards))
print("Average number of interactions for one Q-table", np.mean(average_interactions))
