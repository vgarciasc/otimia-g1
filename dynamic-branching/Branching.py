from math import ceil, fabs, floor
import pdb
import sys
import os
import numpy as np
import cplex as CPX
import cplex.callbacks as CPX_CB
from docplex.mp.model import Model

import traceback
import argparse

from DDQN import DQN
import instance_db
import utils
import plotter

BRANCHING_TYPES = ["Most Infeasible", "Random", "Strong", "Pseudo-cost"]
TRAIN_ON_EVERY = 0
TRAIN_ON_SINGLE = 1
BRANCHING_RL = -1

class BranchCB(CPX_CB.BranchCallback):
    def init(self, _lista):
        self.states_to_process = _lista

        self.times_called = 0
        self.report_count = 0
        self.nodes_count = 0
        self.action_history = []
        self.reward_history = []
        self.optgap_history = []
    
    def branch_most_infeasible(self, node_data):
        x = self.get_values()
        feas = self.get_feasibilities()
        obj = self.get_objective_coefficients()
        objval = self.get_objective_value()

        maxobj = -CPX.infinity
        maxinf = -CPX.infinity

        selected_var = 0
        for j in range(len(x)):
            if feas[j] == self.feasibility_status.infeasible:
                xj_inf = x[j] - floor(x[j])
                if xj_inf > 0.5:
                    xj_inf = 1.0 - xj_inf
                    
                if (xj_inf >= maxinf and (xj_inf > maxinf or fabs(obj[j]) >= maxobj)):
                    selected_var = j
                    maxinf = xj_inf
                    maxobj = fabs(obj[j])
        
        xj_lo = floor(x[selected_var])
        self.make_branch(objval, variables = [(selected_var, "L", xj_lo + 1)], node_data = node_data)
        self.make_branch(objval, variables = [(selected_var, "U", xj_lo    )], node_data = node_data)
        self.nodes_count += 2

    def branch_least_infeasible(self, node_data):
        x = self.get_values()
        feas = self.get_feasibilities()
        obj = self.get_objective_coefficients()
        objval = self.get_objective_value()

        maxobj = -CPX.infinity
        mininf = -CPX.infinity

        selected_var = 0
        for j in range(len(x)):
            if feas[j] == self.feasibility_status.infeasible:
                xj_inf = x[j] - floor(x[j])
                if xj_inf > 0.5:
                    xj_inf = 1.0 - xj_inf
                    
                if (xj_inf <= mininf and (xj_inf < mininf or fabs(obj[j]) >= maxobj)):
                    selected_var = j
                    mininf = xj_inf
                    maxobj = fabs(obj[j])
        
        xj_lo = floor(x[selected_var])
        self.make_branch(objval, variables = [(selected_var, "L", xj_lo + 1)], node_data = node_data)
        self.make_branch(objval, variables = [(selected_var, "U", xj_lo    )], node_data = node_data)
        self.nodes_count += 2
    
    def branch_random(self, node_data):
        x = self.get_values()
        feas = self.get_feasibilities()
        objval = self.get_objective_value()

        feasible_vars = [i for i in range(len(x)) if feas[i] == self.feasibility_status.infeasible]
        if len(feasible_vars) == 0:
            return
        
        selected_var = int(np.random.choice(feasible_vars))
        xj_lo = floor(x[selected_var])
        
        branches = [(selected_var, 'L', xj_lo + 1),
                    (selected_var, 'U', xj_lo)]

        for branch in branches:
            node_data_clone = node_data.copy()
            node_data_clone['branch_history'] = node_data['branch_history'][:]
            node_data_clone['branch_history'].append(branch)

            self.make_branch(objval, variables=[branch], constraints=[], node_data=node_data_clone)
            self.nodes_count += 1

    def branch_strong(self, node_data):
        candidate_idxs = utils.get_candidates(self)
        if len(candidate_idxs) == 0:
            return
        
        sb_scores, _ = utils.get_sb_scores(self, candidate_idxs)
        if len(sb_scores):
            sb_scores = np.asarray(sb_scores)
            branching_var_idx = candidate_idxs[np.argmax(sb_scores)]
            
        objval = self.get_objective_value()
        branching_val = self.get_values(branching_var_idx)

        branches = [(branching_var_idx, 'L', np.floor(branching_val) + 1),
                    (branching_var_idx, 'U', np.floor(branching_val))]

        for branch in branches:
            node_data_clone = node_data.copy()
            node_data_clone['branch_history'] = node_data['branch_history'][:]
            node_data_clone['branch_history'].append(branch)

            self.make_branch(objval, variables=[branch], constraints=[], node_data=node_data_clone)
            self.nodes_count += 1

    def branch_pseudocost(self, node_data):
        objval = self.get_objective_value()
        branches = [self.get_branch(0)[1][0], self.get_branch(1)[1][0]]
        for branch in branches:
            node_data_clone = node_data.copy()
            node_data_clone['branch_history'] = node_data['branch_history'][:]
            node_data_clone['branch_history'].append(branch)

            self.make_branch(objval, variables=[branch], constraints=[], node_data=node_data_clone)
            self.nodes_count += 1
        return
        
    def __call__(self):
        # Counter of how many times the callback was called
        self.times_called += 1
        
        # Getting information about state of node and tree
        last_node_data = self.get_node_data()

        # TODO: Add more information to input
        objval = self.get_objective_value()
        incumbentval = self.get_incumbent_objective_value()
        gap = (objval - incumbentval) / incumbentval
        num_set_variables = len(utils.get_data(self)['branch_history'])
        pc_up, pc_down = zip(*self.get_pseudo_costs())

        # NOTE: Every info should be normalized between 0 and 1!
        state = np.array([[
            self.get_current_node_depth() / ceil(np.log2(5000)), # current depth normalized by maximum depth
            gap, # optimal gap
            np.mean(self.get_feasibilities()), # percentage of feasible variables
            1 - num_set_variables / len(self.get_values()), # percentage of unset variables (DASH)
            np.mean(pc_up), # average pseudo-cost for each variable up
            np.mean(pc_down), # average pseudo-cost for each variable down
            np.mean(self.get_values()), # average number of items in knapsack
        ]])

        if self.branching_strategy == BRANCHING_RL:
            action = dqn.get_action(state)
        else:
            action = self.branching_strategy

        node_data = {'branch_history': [], 'node_id': self.get_node_ID(), 'state': state, 'action': action}
        if last_node_data != None:
            node_data['branch_history'] = last_node_data['branch_history'][:]

        # Debugging
        # print(f"Node_id: {node_id}, state: {state}")
        # print(f'Branching type {BRANCHING_TYPES[action]}, -- Times called: ', self.times_called)
        
        if self.get_num_branches() == 0:
            return
        else:
            if action == 0:
                self.branch_most_infeasible(node_data)
            elif action == 1:
                self.branch_random(node_data)
            elif action == 2:
                self.branch_strong(node_data)
            elif action == 3:
                self.branch_pseudocost(node_data)
            elif action == 4:
                self.branch_least_infeasible(node_data)

        if self.branching_strategy == -1 and last_node_data is not None:
            # Previous state and action are stored in 'node_data' object
            # Because we don't know the reward and next_state until the
            # children nodes are processed
            last_state = last_node_data['state']
            last_action = last_node_data['action']
            last_reward = dqn.calc_reward(last_state, state)
            self.get_MIP_relative_gap()
            self.action_history.append(last_action)
            self.reward_history.append(last_reward)
            self.optgap_history.append(gap)

            dqn.remember(last_state, last_action, last_reward, state, False)
            dqn.replay()
            dqn.target_train()

            # Debugging
            #info = (last_state, last_action, last_reward, state[0], False)
            #print(info)
            #print(f"Reward: {last_reward}")
            # pdb.set_trace()

def init_cplex_model(instance_num, verbose=False):
    # MULTIPLE KNAPSACK
    # v, w, C, K, N = instance_db.get_mkp_instance(instance_num)
    v, w, C, K, N = instance_db.get_bkp_instance_as_mkp(instance_num)
    model = Model('multiple knapsack', log_output=verbose)
    x = model.integer_var_matrix(N, K, name="x")
    for j in range(K):
        model.add_constraint(sum(w[i]*x[i, j] for i in range(N)) <= C[j])
    for i in range(N):
        model.add_constraint(sum(x[i, j] for j in range(K)) <= 10)
    obj_fn = sum(v[i]*x[i,j] for i in range(N) for j in range(K))
    model.set_objective("max", obj_fn)
    
    # BINARY KNAPSACK
    # v, w, C, N = instance_db.get_bkp_instance(instance_num)
    # K = 1
    # C = [C]
    # model = Model('binary knapsack', log_output=verbose)
    # x = model.integer_var_matrix(N, K, name="x")
    # for j in range(K):
    #     model.add_constraint(sum(w[i]*x[i, j] for i in range(N)) <= C[j])
    # for i in range(N):
    #     model.add_constraint(sum(x[i, j] for j in range(K)) <= 1)
    # obj_fn = sum(v[i]*x[i,j] for i in range(N) for j in range(K))
    # model.set_objective("max", obj_fn)

    # Transforming DOCPLEX.MP.MODEL into a CPX.CPLEX object
    filename = "problem.lp"
    model.dump_as_lp(filename)
    cplex = CPX.Cplex(filename)
    os.remove(filename)

    # Displays node information every X nodes
    cplex.parameters.mip.interval.set(1)
    cplex.parameters.mip.limits.nodes.set(5000)

    # Turning off presolving callbacks
    cplex.parameters.preprocessing.presolve.set(0) # Decides whether CPLEX applies presolve during preprocessing to simplify and reduce problems
    cplex.parameters.preprocessing.aggregator.set(0) # Invokes the aggregator to use substitution where possible to reduce the number of rows and columns before the problem is solved. If set to a positive value, the aggregator is applied the specified number of times or until no more reductions are possible.
    cplex.parameters.preprocessing.relax.set(0) # Decides whether LP presolve is applied to the root relaxation in a mixed integer program (MIP). Sometimes additional reductions can be made beyond any MIP presolve reductions that were already done. By default, CPLEX applies presolve to the initial relaxation in order to hasten time to the initial solution.
    cplex.parameters.preprocessing.numpass.set(0) # Limits the number of pre-resolution passes that CPLEX makes during pre-processing. When this parameter is set to a positive value, pre-resolution is applied for the specified number of times or until no further reduction is possible.
    cplex.parameters.mip.cuts.mircut.set(-1) # Decides whether or not to generate MIR cuts (mixed integer rounding cuts) for the problem.

    cplex.parameters.advance.set(0) # If 1 or 2, this parameter specifies that CPLEX should use advanced starting information when it initiates optimization.
    cplex.parameters.preprocessing.qcpduals.set(0) # This parameter determines whether CPLEX preprocesses a quadratically constrained program (QCP) so that the user can access dual values for the QCP.
    cplex.parameters.preprocessing.qpmakepsd.set(0) # Decides whether CPLEX will attempt to reformulate a MIQP or MIQCP model that contains only binary variables. When this feature is active, adjustments will be made to the elements of a quadratic matrix that is not nominally positive semi-definite (PSD, as required by CPLEX for all QP and most QCP formulations), to make it PSD, and CPLEX will also attempt to tighten an already PSD matrix for better numerical behavior.
    cplex.parameters.preprocessing.qtolin.set(0) # This parameter switches on or off linearization of the quadratic terms in the objective function of a quadratic program (QP) or of a mixed integer quadratic program (MIQP) during preprocessing.
    cplex.parameters.preprocessing.repeatpresolve.set(0) # Specifies whether to re-apply presolve, with or without cuts, to a MIP model after processing at the root is otherwise complete.
    cplex.parameters.preprocessing.dual.set(0) # Decides whether the CPLEX pre-solution should pass the primal or dual linear programming problem to the linear programming optimization algorithm.
    cplex.parameters.preprocessing.fill.set(0) # Limits number of variable substitutions by the aggregator. If the net result of a single substitution is more nonzeros than this value, the substitution is not made.
    cplex.parameters.preprocessing.coeffreduce.set(0) # Decides how coefficient reduction is used. Coefficient reduction improves the objective value of the initial (and subsequent) LP relaxations solved during branch and cut by reducing the number of non-integral vertices. By default, CPLEX applies coefficient reductions during preprocessing of a model.
    cplex.parameters.preprocessing.boundstrength.set(0) # Decides whether to apply bound strengthening in mixed integer programs (MIPs). Bound strengthening tightens the bounds on variables, perhaps to the point where the variable can be fixed and thus removed from consideration during branch and cut.
    cplex.parameters.preprocessing.dependency.set(0) # Decides whether to activate the dependency checker. If on, the dependency checker searches for dependent rows during preprocessing. If off, dependent rows are not identified.
    cplex.parameters.preprocessing.folding.set(0) # Decides whether folding will be automatically executed, during the preprocessing phase, in a LP model.
    cplex.parameters.preprocessing.symmetry.set(0) # Decides whether symmetry breaking reductions will be automatically executed, during the preprocessing phase, in either a MIP or LP model.
    cplex.parameters.preprocessing.sos1reform.set(-1) # This parameter allows you to control the reformulation of special ordered sets of type 1 (SOS1), which can be applied during the solution process of problems containing these sets.
    cplex.parameters.preprocessing.sos2reform.set(-1) # This parameter allows you to control the reformulation of special ordered sets of type 2 (SOS2), which can be applied during the solution process of problems containing these sets.

    cplex.parameters.mip.strategy.variableselect.set(3) # Pseudo-cost branching: DO NOT CHANGE!

    num_vars = cplex.variables.get_num()

    # Registering the branching callback
    states_to_process = []
    branch_callback = cplex.register_callback(BranchCB)
    branch_callback.init(states_to_process)
    branch_callback.ordered_var_idx_lst = list(range(num_vars))
    branch_callback.c = cplex
    branch_callback.num_infeasible_left = np.zeros(num_vars)
    branch_callback.num_infeasible_right = np.zeros(num_vars)
    branch_callback.times_called = 0
    branch_callback.THETA = 200
    branch_callback.max_iterations = 500
    branch_callback.model = None

    return cplex, branch_callback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamic Branching')
    parser.add_argument('--episodes', help="How many episodes run?", required=True, type=int)
    parser.add_argument('--branching_strategy', help="Which branching strategy to use?", required=True, type=int)
    parser.add_argument('--training_scheme', help='Which training scheme to use? 0 is train on every instance, 1 is train on single instance', required=False, default=-1, type=int)
    parser.add_argument('--single_instance', help='Which single instance to run?', required=False, default=-1, type=int)
    parser.add_argument('--execution_name', help='What is the execution name?', required=False, default="", type=str)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    episodes = args['episodes']

    if args['training_scheme'] == TRAIN_ON_EVERY:
        instances_to_train = [i for i, _ in enumerate(instance_db.get_bkp_filenames())]
    elif args['training_scheme'] == TRAIN_ON_SINGLE:
        instances_to_train = [args['single_instance']]
        
    dqn = DQN(n_actions=len(BRANCHING_TYPES), n_inputs=7)
    action_history = []
    reward_history = []
    optgap_history = []

    cplex_history = []

    for episode in range(episodes):
        for instance_num in instances_to_train:
            cplex, branch_callback = init_cplex_model(instance_num=instance_num, verbose=args['verbose'])
            branch_callback.branching_strategy = args['branching_strategy']

            cplex.solve()
            
            action_history = np.append(action_history, branch_callback.action_history)
            reward_history = np.append(reward_history, branch_callback.reward_history)
            optgap_history = np.append(optgap_history, branch_callback.optgap_history)
            cplex_history.append((
                instance_db.get_bkp_filenames()[instance_num],
                branch_callback.nodes_count, 
                cplex.solution.MIP.get_mip_relative_gap(),
                cplex.solution.MIP.get_best_objective()))

            plotter.plot_action_history(action_history, BRANCHING_TYPES, episode, args['execution_name'])
            plotter.plot_reward_history(reward_history, episode, args['execution_name'])
            plotter.plot_generic(dqn.loss_history, "DQN Loss", episode, args['execution_name'])
            plotter.plot_generic(optgap_history, "Optimality Gap", episode, args['execution_name'])

    for filename, nodes_opened, gap, best_objective in cplex_history:
        print(f"{filename}")
        print(f"-- Nodes opened: {nodes_opened}")
        print(f"-- Optimal gap: {gap}")
        print(f"-- Best objective: {best_objective}")
        print()

    pdb.set_trace()
    print('Done')