import numpy as np
import cplex as CPX
import cplex.callbacks as CPX_CB
from operator import itemgetter

LP_OPTIMAL = 1
LP_INFEASIBLE = 3
LP_ABORT_IT_LIM = 10
EPSILON = 1e-6
CPX_DEFAULT = 0
CPX_PC = 1
CPX_SB = 2
BS_SB = 3
BS_PC = 4
BS_SB_PC = 5
BS_SB_ML_SVMRank = 6
BS_SB_ML_NN = 7

def get_data(context):
    node_data = context.get_node_data()
    
    if node_data is None:
        node_data = {'branch_history': []}

    return node_data

def get_clone(context):
    cclone = CPX.Cplex(context.c)

    node_data = get_data(context)
    apply_branch_history(cclone, node_data['branch_history'])

    return cclone

def apply_branch_history(c, branch_history):
    for b in branch_history:
        b_var_idx = b[0]
        b_type = b[1]
        b_val = b[2]

        if b_type == 'L':
            c.variables.set_lower_bounds(b_var_idx, b_val)
        elif b_type == 'U':
            c.variables.set_upper_bounds(b_var_idx, b_val)

def disable_output(c):
    c.set_log_stream(None)
    c.set_error_stream(None)
    c.set_warning_stream(None)
    c.set_results_stream(None)

def solve_as_lp(c, max_iterations=50):
    disable_output(c)
    # Create LP for the input MIP
    c.set_problem_type(c.problem_type.LP)
    # Set the maximum number of iterations for solving the LP
    if max_iterations is not None:
        c.parameters.simplex.limits.iterations = max_iterations

    c.solve()
    status, objective, dual_values = None, None, None
    status = c.solution.get_status()
    if status == LP_OPTIMAL or status == LP_ABORT_IT_LIM:
        objective = c.solution.get_objective_value()
        dual_values = c.solution.get_dual_values()

    return status, objective, dual_values

def get_branch_solution(context, cclone, var_idx, bound_type):
    value = context.get_values(var_idx)

    get_bounds = None
    set_bounds = None
    new_bound = None
    if bound_type == 'L':
        get_bounds = context.get_lower_bounds
        set_bounds = cclone.variables.set_lower_bounds
        new_bound = np.floor(value) + 1
    elif bound_type == 'U':
        get_bounds = context.get_upper_bounds
        set_bounds = cclone.variables.set_upper_bounds
        new_bound = np.floor(value)

    original_bound = get_bounds(var_idx)

    set_bounds(var_idx, new_bound)
    status, objective, _ = solve_as_lp(cclone)
    set_bounds(var_idx, original_bound)

    return status, objective

def get_candidates(context):
    """Find candidate variables at the current node in the B&B tree
    for branching.
    """
    pseudocosts = context.get_pseudo_costs(context.ordered_var_idx_lst)
    values = context.get_values(context.ordered_var_idx_lst)

    up_frac = np.ceil(values) - values
    down_frac = values - np.floor(values)

    # Find scores
    scores = [(uf * df * pc[0] * pc[1], vidx)
              for vidx, (pc, uf, df) in enumerate(zip(pseudocosts, up_frac, down_frac))]

    # Sort scores in descending order
    scores = sorted(scores, key=itemgetter(0), reverse=True)

    # Select candidates based on sorted scores
    num_candidates = 10
    # num_candidates = 10 if context.branch_strategy != BS_PC else 1
    candidate_idxs = []
    ranked_var_idx_lst = [i[1] for i in scores]
    for var_idx in ranked_var_idx_lst:
        if len(candidate_idxs) == num_candidates:
            break

        value = values[var_idx]
        if not abs(value - round(value)) <= EPSILON:
            candidate_idxs.append(var_idx)

    return candidate_idxs

def get_sb_scores(context, candidate_idxs):
    cclone = get_clone(context)
    status, parent_objective, dual_values = solve_as_lp(cclone, max_iterations=context.max_iterations)

    sb_scores = []
    if status == LP_OPTIMAL or status == LP_ABORT_IT_LIM:
        context.curr_node_dual_values = np.asarray(dual_values)
        for var_idx in candidate_idxs:
            upper_status, upper_objective = get_branch_solution(context, cclone, var_idx, 'L')
            lower_status, lower_objective = get_branch_solution(context, cclone, var_idx, 'U')

            # Infeasibility leads to higher score as it helps in pruning the tree
            if upper_status == 3:
                upper_objective = 1e6
                context.num_infeasible_right[var_idx] += 1
            if lower_status == 3:
                lower_objective = 1e6
                context.num_infeasible_left[var_idx] += 1

            # Calculate deltas
            delta_upper = max(upper_objective - parent_objective, EPSILON)
            delta_lower = max(lower_objective - parent_objective, EPSILON)

            # Calculate sb score
            sb_score = delta_lower * delta_upper
            sb_scores.append(sb_score)

    else:
        print("Root LP infeasible...")

    return sb_scores, cclone

def get_data(context):
    node_data = context.get_node_data()
    if node_data is None:
        node_data = {'branch_history': []}

    return node_data