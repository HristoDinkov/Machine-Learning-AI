import random
import time
import matplotlib.pyplot as plt
from environment import DungeonSimulator
from agent import Drunkard
from agent import Accountant
from agent import Gambler
from agent import DeepGambler

learning_rate = 0.1
discount = 0.95
iterations = 10000


def main():

    agent = DeepGambler()            # Init the agent
    dungeon = DungeonSimulator()    # Init the dungeon
    dungeon.reset()
    total_reward = 0                # Keep the score
    steps = []
    rewards = []
    performance = []
    last_total = 0

    for step in range(iterations + 1):
        old_state = dungeon.state                           # Store current state
        action = agent.get_next_action(old_state)           # Query agent for the next action
        new_state, reward = dungeon.take_action(action)     # Take action, get the reward
        agent.update(old_state, new_state, action, reward)  # Update the agent

        total_reward += reward                              # Add to total score

        rewards.append(total_reward)

        if step % 250 == 0:                                 # Print the score every 250 iterations
            performance.append((total_reward - last_total) / 250.0)
            steps.append(step)
            last_total = total_reward
            print(f'Step: {step}    Performance: {performance[-1]}')

        time.sleep(0.0001)

    plt.plot(steps, performance)
    plt.show()


if __name__ == "__main__":
    main()


