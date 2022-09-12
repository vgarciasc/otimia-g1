import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BRANCHING_TYPES = ["Most Infeasible", "Random", "Strong", "Pseudo-cost"]
from instance_db import get_bkp_filenames_test

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
    plt.savefig(f"data/{execution_name}_actions.png")
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
    plt.savefig(f"data/{execution_name}_rewards")
    # plt.show()
    plt.close('all')

def plot_generic(array, s, execution_name):
    plt.title(s)
    for i in range(len(array)):
        if i % 100 == 0:
            plt.plot(i, np.mean(array[i-100:i]), "bo")
    plt.xlabel("Iteration")
    plt.legend()
    plt.savefig(f"data/{execution_name}_{s}")
    plt.close('all')

def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def plot_optimality_gap(model):
    # instance_name = get_bkp_filenames_hard(instance_num)
    instance_num = 0
    instance_name = get_bkp_filenames_test()[instance_num]

    rl_optgap = pd.read_csv(f"data/{model}_TESTING_instance_{instance_num}_optgap_history_test.csv", 
        names=["iteration", "optgap"], header=1)["optgap"].to_numpy()
    pc_optgap = pd.read_csv(f"data/pseudocost_TESTING_instance_{instance_num}_optgap_history_test.csv", 
        names=["iteration", "optgap"], header=1)["optgap"].to_numpy()
    sb_optgap = pd.read_csv(f"data/strong_TESTING_instance_{instance_num}_optgap_history_test.csv", 
        names=["iteration", "optgap"], header=1)["optgap"].to_numpy()
    rd_optgap = pd.read_csv(f"data/random_TESTING_instance_{instance_num}_optgap_history_test.csv", 
        names=["iteration", "optgap"], header=1)["optgap"].to_numpy()
    mf_optgap = pd.read_csv(f"data/most_fractional_TESTING_instance_{instance_num}_optgap_history_test.csv", 
        names=["iteration", "optgap"], header=1)["optgap"].to_numpy()
    
    rl_optgap = moving_average(rl_optgap, 5)
    pc_optgap = moving_average(pc_optgap, 5)
    sb_optgap = moving_average(sb_optgap, 5)
    rd_optgap = moving_average(rd_optgap, 5)
    mf_optgap = moving_average(mf_optgap, 5)

    plt.plot(pc_optgap, label="Pseudo-cost branching")
    plt.plot(sb_optgap, label="Strong branching")
    plt.plot(rd_optgap, label="Random branching")
    plt.plot(mf_optgap, label="Most fractional branching")
    plt.plot(rl_optgap, label="RL branching", linewidth=3)
    plt.xlabel("Iterations")
    plt.ylabel("Optimal gap")
    plt.title(f"Comparing different branching strategies in Instance {instance_name}")
    plt.legend()
    plt.show()

def plot_actions_in_test_instance(filename, title):
    actions = pd.read_csv(filename, names=["iteration", "action"], header=1)["action"].to_numpy()
    plt.plot(actions, linewidth=2, color='blue')
    # plt.scatter(range(len(actions)), actions, color='red')
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Action idx")
    plt.yticks(np.arange(len(BRANCHING_TYPES)), BRANCHING_TYPES)
    plt.show()

if __name__ == "__main__":
    # plot_optimality_gap("boltzmann_depth_gap_ep29")
    # plot_actions_in_test_instance(
    #     "data/boltzmann_depth_gap_ep29_TESTING_instance_0_action_history_test.csv", 
    #     "Boltzmann Depth Gap on instance \n'large_scale/knapPI_1_100_1000_1'")
        
    # plot_actions_in_test_instance(
    #     "data/boltzmann_depth_gap_ep29_TESTING_instance_1_action_history_test.csv", 
    #     "Boltzmann Depth Gap on instance \n'all/probT1_1W_R50_T002_M010_N0020_seed02'")
        
    # plot_actions_in_test_instance(
    #     "data/boltzmann_depth_gap_ep29_TESTING_instance_2_action_history_test.csv", 
    #     "Boltzmann Depth Gap on instance \n'all/probT1_1W_R50_T002_M010_N0020_seed01'")

    # plot_actions_in_test_instance(
    #     "data/punitive_boltzmann_depth_gap_ep28_TESTING_instance_4_hard_action_history_test.csv", 
    #     "Punitive Boltzmann Depth Gap on instance \n'n_800_c_1000000_g_14_f_0.1_eps_0.1_s_100'")

    plot_actions_in_test_instance(
        "data/punitive_boltzmann_depth_gap_ep28_TESTING_instance_5_hard_action_history_test.csv", 
        "Punitive Boltzmann Depth Gap on instance \n'n_1000_c_1000000_g_14_f_0.2_eps_0.1_s_100'")