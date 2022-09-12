from math import fabs, floor
from docplex.mp.model import Model
import cplex as CPX
import cplex.callbacks as CPX_CB
import numpy as np
import pdb

import instance_db

class MyBranch(CPX_CB.BranchCallback):
    def __call__(self):
        # Counter of how many times the callback was called
        self.times_called += 1

        br_type = self.get_branch_type()
        if br_type == self.branch_type.SOS1 or br_type == self.branch_type.SOS2:
            return

        # Getting information about state of node and tree
        x = self.get_values()
        objval = self.get_objective_value()
        obj    = self.get_objective_coefficients()
        feas   = self.get_feasibilities()
        
        maxobj = -CPX.infinity
        maxinf = -CPX.infinity
        bestj  = -1

        branching = "most_fractional"
        # branching = "random"

        if branching == "most_fractional":
            # MOST FRACTIONAL BRANCHING
            for j in range(len(x)):
                if feas[j] == self.feasibility_status.infeasible:
                    xj_inf = x[j] - floor(x[j])
                    if xj_inf > 0.5:
                        xj_inf = 1.0 - xj_inf
                        
                    if (xj_inf >= maxinf and (xj_inf > maxinf or fabs(obj[j]) >= maxobj)):
                        bestj = j
                        maxinf = xj_inf
                        maxobj = fabs(obj[j])
        elif branching == "random":
            # RANDOM BRANCHING
            feasible_vars = [i for i in range(len(x)) if feas[i] == self.feasibility_status.infeasible]
            if len(feasible_vars) == 0:
                return
            bestj = int(np.random.choice(feasible_vars))
        
        if bestj < 0:
            return
    
        # Making the branching
        xj_lo = floor(x[bestj])
        self.make_branch(objval, variables = [(bestj, "L", xj_lo + 1)], node_data = (bestj, xj_lo, "UP"))
        self.make_branch(objval, variables = [(bestj, "U", xj_lo)], node_data = (bestj, xj_lo, "DOWN"))

if __name__ == "__main__":
    v, w, C, K, N = instance_db.get_instance(2)
    m = Model('multiple knapsack', log_output=True)

    # If set to X, information will be displayed every X iterations
    # m.parameters.mip.interval.set(1)

    # Turning off presolving callbacks
    m.parameters.preprocessing.presolve.set(0)
    m.parameters.preprocessing.aggregator.set(0)
    m.parameters.preprocessing.reduce.set(0)
    m.parameters.preprocessing.relax.set(0)
    m.parameters.preprocessing.numpass.set(0)

    # Registering the branching callback
    branch_instance = m.register_callback(MyBranch)
    branch_instance.times_called = 0

    # Adding variables
    x = m.integer_var_matrix(N, K, name="x")

    # Adding constraints
    for j in range(K):
        m.add_constraint(sum(w[i]*x[i, j] for i in range(N)) <= C[j])
    for i in range(N):
        m.add_constraint(sum(x[i, j] for j in range(K)) <= 10)

    # Setting up the objective function
    obj_fn = sum(v[i]*x[i,j] for i in range(N) for j in range(K))
    m.set_objective("max", obj_fn)

    # Displaying info
    m.print_information()

    # Solving... Should take a while
    sol = m.solve()

    # Printing solution
    # m.print_solution()

    # Displaying final information
    print(f"branch_instance.times_called: {branch_instance.times_called}")
    print("max: {}".format(m.solution.get_objective_value()))
    if sol is None:
        print("Infeasible")

    print("==> Done.")
    pdb.set_trace()