import numpy as np
import matplotlib.pyplot as plt

def plot_action_history(action_history, branching_types, episode):
    branching_strats_history = []
    for i, branching in enumerate(branching_types):
        branching_strats_history.append([])

        action_counter = 0
        total_actions = 0
        for action in action_history:
            action_counter += 1 if (action == i) else 0
            total_actions += 1

            branching_strats_history[i].append(action_counter / total_actions)
    
        plt.plot(branching_strats_history[i], label=branching)
    
    plt.title("Branching decisions over time")
    plt.ylabel("Frequency")
    plt.xlabel("Iteration")
    plt.legend()
    #plt.savefig("actions"+str(episode)+".png")
    plt.savefig(f"actions_{episode}.png")
    # plt.show()
    plt.close('all')

def plot_reward_history(reward_history, episode):
    average_reward = []
    total_reward = 0

    for i, reward in enumerate(reward_history):
        total_reward += reward
        average_reward.append(total_reward / (i + 1))
    
    plt.title("Reward over time")
    plt.plot(average_reward, color='blue', label='Average reward')
    # plt.plot(reward_history, color='cyan', linestyle='dashed', label='Absolute reward')
    plt.xlabel("Iteration")
    plt.legend()
    plt.savefig(f"rewards_{episode}")
    # plt.show()
    plt.close('all')

def plot_generic(array, s, episode):
  plt.title(s)
  plt.plot(array)
  plt.xlabel("Iteration")
  plt.legend()
  plt.savefig(f"{s}_{episode}")
  plt.close('all')
    