import numpy as np
import matplotlib.pyplot as plt

def plot_action_history(action_history, branching_types, execution_name):
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
    plt.savefig(f"data/actions_{execution_name}.png")
    # plt.show()
    plt.close('all')

def plot_reward_history(reward_history, execution_name):
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
    plt.savefig(f"data/rewards_{execution_name}")
    # plt.show()
    plt.close('all')

def plot_generic(array, s, execution_name):
  plt.title(s)
  for i in range(len(array)):
    if i % 100 == 0:
      plt.plot(i, np.mean(array[i-100:i]), "bo")
  plt.xlabel("Iteration")
  plt.legend()
  plt.savefig(f"data/{s}_{execution_name}")
  plt.close('all')
    