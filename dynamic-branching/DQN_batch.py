# utils imports
import pdb
import time
import numpy as np
import pandas as pd
import random
from collections import deque
from math import fabs, floor
import os
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "plaidml.bridge.keras"

# Gym imports
import gym
from gym import Env
from gym.spaces import Discrete, Box

# Keras imports
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# cplex imports
from docplex.mp.model import Model
import cplex as CPX
import cplex.callbacks as CPX_CB
from cplex.callbacks import SolutionStrategy, MIPCallback, BranchCallback

import instance_db
import plotter

BRANCHING_TYPES = ["Most Fractional", "Random", "Minimum Infeasibility", "Maximum Infeasibility", "Pseudo-cost", "Strong", "Pseudo-reduced-cost"]

class DQN:
    def __init__(self, memory_size=5*10**5, batch_size=32, gamma=0.99,
        exploration_max=1.0, exploration_min=0.01, exploration_decay=0.99999,
        learning_rate=0.001, tau=0.125):
        
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.learning_rate = learning_rate
        self.tau = tau

        self.action_space = Discrete(len(BRANCHING_TYPES))
        self.observation_space = Box(low=np.array([0]), high=np.array([10**5]), dtype=np.float32)

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.loss_history = []
        self.fit_count = 0

        self.nodes_queue = []

    def create_model(self):
        _model = Sequential()
        state_shape = self.observation_space.shape
        _model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        # _model.add(Dense(48, activation="relu"))
        _model.add(Dense(24, activation="relu"))
        _model.add(Dense(self.action_space.n))
        _model.compile(loss="mse", 
                       optimizer=Adam(learning_rate=self.learning_rate))

        return _model

    def get_action(self, state):
        self.exploration_max *= self.exploration_decay
        self.exploration_max = max(self.exploration_min, self.exploration_max)
        if np.random.random() < self.exploration_max:
            return self.action_space.sample()
        # q_values = self.model.predict(state, verbose=0)
        q_values = self.model(state).numpy()
        best_action = np.argmax(q_values[0])
        return best_action

    def remember(self, state, action, reward, next_state, done):
        state = np.array([state['objval']])
        next_state = np.array([next_state['objval']])
        
        self.memory.append([state, action, reward, next_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 
        
        samples = random.sample(self.memory, self.batch_size)
        states = np.array([sample[0] for sample in samples])
        states, actions, rewards, next_states, dones = zip(*samples)
        # targets = self.target_model.predict(states, batch_size=self.batch_size, verbose=False).numpy()[0]
        # Qs_future = [max(self.target_model(i).numpy()[0]) for i in next_states]
        targets = []
        for state, action, reward, next_state, done in samples:
            # target = self.target_model.predict(state, verbose=0)
            target = self.target_model(state).numpy()[0]
            if done:
                target[action] = reward
            else:
                # Q_future = max(self.target_model.predict(next_state, verbose=0)[0])
                Q_future = max(self.target_model(next_state)[0])
                target[action] = reward + Q_future * self.gamma
            # targets.append(target[0])
            targets.append(target)
            # self.model.fit(state, target, epochs=1, verbose=0)
        # targets = np.delete(targets, 0, axis=0)
        states = np.array(states)
        targets = np.array(targets)
        self.loss_history.append(self.model.fit(states, targets, verbose=0).history['loss'][0])

        # self.fit_count += 1
        # if self.fit_count % 500 == 0 and self.fit_count > 0:
        #     plt.plot(self.loss_history)
        #     plt.savefig("loss"+str(self.fit_count)+".png")
        #     plt.close()
        #     pd.DataFrame(self.loss_history).to_csv("saved/loss.csv")

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
    
    def calc_reward(self, state, next_state):
        # Not sure if it's objval we want. Should double-check
        if next_state['objval'] < state['objval']:
            return 100
        if next_state['objval'] > state['objval']:
            return 0
        else: 
            return 1

    def save_model(self, fn):
        self.model.save(fn)

class BranchCB(CPX_CB.BranchCallback):
    def init(self, _lista):
        self.states_to_process = _lista
        self.times_called = 0
        self.report_count = 0
        self.branches_count = 0
        self.action_history = []
        self.reward_history = []
        self.optgap_history = []
    
    def __call__(self):
        # Counter of how many times the callback was called
        self.times_called += 1
        
        # Getting information about state of node and tree
        x = self.get_values()

        objval = self.get_objective_value()
        best_objval = self.get_best_objective_value()

        obj    = self.get_objective_coefficients()
        feas   = self.get_feasibilities()

        node_id = self.get_node_ID() # node id of the current node
        last_node_data = self.get_node_data()
        incumbentval = self.get_incumbent_objective_value() # value of the incumbent solution
        cutoff = self.get_cutoff() # cutoff value
        
        state = {'best_objval':   best_objval, # best objective function value, i.e. the best known dual bound
                 'objval':        objval,  # function value at the current node
                 'incumbentval':  incumbentval, # value of the incumbent/current solution - i.e. the best known primal bound
                 'cutoff':        cutoff, # cutoff value, i.e. the best known primal bound + 1
                 'gap':           (objval - incumbentval) / incumbentval}
        action = dqn.get_action(state)

        node_data = {'node_id': self.get_node_ID(), 'state': state, 'action': action}

        # Debugging
        # print(f"Node_id: {node_id}, state: {state}")
        # print(f'Branching type {BRANCHING_TYPES[action]}, -- Times called: ', self.times_called)

        selected_var = -1
        if action == 0: # Branch on variable with most fractional value (nearest to 0.5)
            maxobj = -CPX.infinity
            maxinf = -CPX.infinity

            for j in range(len(x)):
                if feas[j] == self.feasibility_status.infeasible:
                    xj_inf = x[j] - floor(x[j])
                    if xj_inf > 0.5:
                        xj_inf = 1.0 - xj_inf
                        
                    if (xj_inf >= maxinf and (xj_inf > maxinf or fabs(obj[j]) >= maxobj)):
                        selected_var = j
                        maxinf = xj_inf
                        maxobj = fabs(obj[j])
        elif action == 1: # Branch on random variable
            feasible_vars = [i for i in range(len(x)) if feas[i] == self.feasibility_status.infeasible]
            if len(feasible_vars) == 0:
                return
            selected_var = int(np.random.choice(feasible_vars))
            if selected_var < 0:
                return
        elif action == 2:
            cplex.parameters.mip.strategy.variableselect = -1 # Branch on variable with minimum infeasibility
        elif action == 3:
            cplex.parameters.mip.strategy.variableselect = 1 # Branch on variable with maximum infeasibility
        elif action == 4:
            cplex.parameters.mip.strategy.variableselect = 2 # Branch based on pseudo costs
        elif action == 5:
            cplex.parameters.mip.strategy.variableselect = 3 # Strong branching
        elif action == 6:
            cplex.parameters.mip.strategy.variableselect = 4 # Branch based on pseudo reduced costs

        # Making the branching  
        if action == 0 or action == 1:
            xj_lo = floor(x[selected_var])

            self.make_branch(objval, variables = [(selected_var, "U", xj_lo    )], node_data = node_data)
            self.make_branch(objval, variables = [(selected_var, "L", xj_lo + 1)], node_data = node_data)
        else:
            self.make_branch(objval, node_data = node_data)
            self.make_branch(objval, node_data = node_data)

        if last_node_data is not None:
            last_state = last_node_data['state']
            last_action = last_node_data['action']
            last_reward = dqn.calc_reward(last_state, state)

            self.action_history.append(last_action)
            self.reward_history.append(last_reward)

            # Debugging
            info = (last_state, last_action, last_reward, state['objval'], False)
            print(info)
            print(f"Reward: {last_reward}")
            # pdb.set_trace()

            dqn.remember(last_state, last_action, last_reward, state, False)
            dqn.replay()
            dqn.target_train()

        # self.report_count += 1
        # if self.report_count % 500 == 0 and self.report_count > 0:
        #     pd.DataFrame(self.states_to_process).to_csv('saved/states_to_process.csv')  

def init_cplex_model():
    v, w, C, K, N = instance_db.get_instance(1)

    cplex = Model('multiple knapsack', log_output=False)
    cplex.data = -1

    # If set to X, information will be displayed every X iterations
    cplex.parameters.mip.interval.set(1)

    # Turning off presolving callbacks
    cplex.parameters.preprocessing.presolve.set(0) # Decides whether CPLEX applies presolve during preprocessing to simplify and reduce problems
    cplex.parameters.preprocessing.aggregator.set(0) # Invokes the aggregator to use substitution where possible to reduce the number of rows and columns before the problem is solved. If set to a positive value, the aggregator is applied the specified number of times or until no more reductions are possible.
    cplex.parameters.preprocessing.reduce.set(0) # 
    # cplex_m.parameters.preprocessing.linear.set(0) # Decides whether linear or full reductions occur during preprocessing. If only linear reductions are performed, each variable in the original model can be expressed as a linear form of variables in the presolved model. This condition guarantees, for example, that users can add their own custom cuts to the presolved model.
    cplex.parameters.preprocessing.relax.set(0) # Decides whether LP presolve is applied to the root relaxation in a mixed integer program (MIP). Sometimes additional reductions can be made beyond any MIP presolve reductions that were already done. By default, CPLEX applies presolve to the initial relaxation in order to hasten time to the initial solution.
    cplex.parameters.preprocessing.numpass.set(0) # Limits the number of pre-resolution passes that CPLEX makes during pre-processing. When this parameter is set to a positive value, pre-resolution is applied for the specified number of times or until no further reduction is possible.

    # cplex.parameters.advance.set(0) # If 1 or 2, this parameter specifies that CPLEX should use advanced starting information when it initiates optimization.
    # cplex.parameters.preprocessing.qcpduals.set(0) # This parameter determines whether CPLEX preprocesses a quadratically constrained program (QCP) so that the user can access dual values for the QCP.
    # cplex.parameters.preprocessing.qpmakepsd.set(0) # Decides whether CPLEX will attempt to reformulate a MIQP or MIQCP model that contains only binary variables. When this feature is active, adjustments will be made to the elements of a quadratic matrix that is not nominally positive semi-definite (PSD, as required by CPLEX for all QP and most QCP formulations), to make it PSD, and CPLEX will also attempt to tighten an already PSD matrix for better numerical behavior.
    # cplex.parameters.preprocessing.qtolin.set(0) # This parameter switches on or off linearization of the quadratic terms in the objective function of a quadratic program (QP) or of a mixed integer quadratic program (MIQP) during preprocessing.
    # cplex.parameters.preprocessing.repeatpresolve.set(0) # Specifies whether to re-apply presolve, with or without cuts, to a MIP model after processing at the root is otherwise complete.
    # cplex.parameters.preprocessing.dual.set(0) # Decides whether the CPLEX pre-solution should pass the primal or dual linear programming problem to the linear programming optimization algorithm.
    # cplex.parameters.preprocessing.fill.set(0) # Limits number of variable substitutions by the aggregator. If the net result of a single substitution is more nonzeros than this value, the substitution is not made.
    # cplex.parameters.preprocessing.coeffreduce.set(0) # Decides how coefficient reduction is used. Coefficient reduction improves the objective value of the initial (and subsequent) LP relaxations solved during branch and cut by reducing the number of non-integral vertices. By default, CPLEX applies coefficient reductions during preprocessing of a model.
    # cplex.parameters.preprocessing.boundstrength.set(0) # Decides whether to apply bound strengthening in mixed integer programs (MIPs). Bound strengthening tightens the bounds on variables, perhaps to the point where the variable can be fixed and thus removed from consideration during branch and cut.
    # cplex.parameters.preprocessing.dependency.set(0) # Decides whether to activate the dependency checker. If on, the dependency checker searches for dependent rows during preprocessing. If off, dependent rows are not identified.
    # cplex.parameters.preprocessing.folding.set(0) # Decides whether folding will be automatically executed, during the preprocessing phase, in a LP model.
    # cplex.parameters.preprocessing.symmetry.set(0) # Decides whether symmetry breaking reductions will be automatically executed, during the preprocessing phase, in either a MIP or LP model.
    # cplex.parameters.preprocessing.sos1reform.set(-1) # This parameter allows you to control the reformulation of special ordered sets of type 1 (SOS1), which can be applied during the solution process of problems containing these sets.
    # cplex.parameters.preprocessing.sos2reform.set(-1) # This parameter allows you to control the reformulation of special ordered sets of type 2 (SOS2), which can be applied during the solution process of problems containing these sets.
    # cplex.parameters.mip.cuts.mircut(-1) # Decides whether or not to generate MIR cuts (mixed integer rounding cuts) for the problem.

    # Registering the branching callback
    states_to_process = []
    branch_callback = cplex.register_callback(BranchCB)
    branch_callback.init(states_to_process)

    # Adding variables
    x = cplex.integer_var_matrix(N, K, name="x")

    # Adding constraints
    for j in range(K):
        cplex.add_constraint(sum(w[i]*x[i, j] for i in range(N)) <= C[j])
    for i in range(N):
        cplex.add_constraint(sum(x[i, j] for j in range(K)) <= 10)

    # Setting up the objective function
    obj_fn = sum(v[i]*x[i,j] for i in range(N) for j in range(K))
    cplex.set_objective("max", obj_fn)

    # Displaying info
    # m.print_information()

    # Printing solution
    # m.print_solution()

    # Displaying final information
    # cplex_m.print_information()

    return cplex, branch_callback

if __name__ == "__main__":
    episodes = 1
    
    dqn = DQN()

    for episode in range(episodes):
        cplex, branch_callback = init_cplex_model()
        cplex.solve()
        plotter.plot_action_history(branch_callback.action_history, BRANCHING_TYPES)
        plotter.plot_reward_history(branch_callback.reward_history)