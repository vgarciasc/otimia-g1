import math
import pdb
import numpy as np

class BinaryKnapsackProblem:
    def __init__(self, w, p, C):
        self.w = np.array(w)
        self.p = np.array(p)
        self.C = C
        self.N = len(w)
    
    # lagrangian suproblem can be solved by inspection:
    # item is inserted into knapsack if (p_i - u w_i) > 0
    def solve_lagrangian_subproblem(self, u):
        x = [1 if (self.p[i] - u * self.w[i] >= 0) else 0 for i in range(self.N)]
        x = np.array(x)
        z = u * self.C + x @ (self.p - u * self.w)
        return x, z

    # removes items from knapsack from least dense to most dense
    # until capacity C is reached
    def apply_lagrangian_heuristic_greedy(self, x):
        x = np.copy(x)

        while x @ self.w > self.C:
            less_dense = -1
            min_density = math.inf

            for i in range(self.N):
                if x[i] == 1:
                    density = self.p[i] / self.w[i]
                    if density < min_density:
                        less_dense = i
                        min_density = density
            
            x[less_dense] = 0
        
        return x, self.evaluate_solution(x)
    
    # removes items from knapsack with a probability proportional to
    # their density, until capacity C is reached
    def apply_lagrangian_heuristic_densityprob(self, x):
        x = np.copy(x)

        while x @ self.w > self.C:
            candidates, inverse_density = zip(*[(i, self.w[i] / self.p[i]) for i in range(self.N) if x[i] == 1])
            inverse_density /= np.sum(inverse_density)
            
            selected_for_removal = np.random.choice(candidates, p=inverse_density)
            x[selected_for_removal] = 0
        
        return x, self.evaluate_solution(x)
    
    def evaluate_solution(self, x):
        return x @ self.p
    
    def __str__(self):
        str = ""
        str += f"Knapsack Problem with {self.N} items.\n"
        str += f"- weights: {self.w}\n"
        str += f"- profits: {self.p}\n"
        str += f"- capacity: {self.C}\n"
        return str

if __name__ == "__main__":
    w = [4, 2, 5, 4, 5, 1, 3, 5]
    p = [10, 5, 18, 12, 15, 1, 2, 8]
    C = 15
    N = len(w)

    kp = BinaryKnapsackProblem(w, p, C)
    print(kp)

    u = np.ones(N)
    x, z_u = kp.solve_lagrangian_subproblem(u)
    print(f"Lagrangian multipliers: {u}")
    print(f"Lagrangian subproblem solution: {x}")
    print(f"Dual bound: {z_u}")
    print("")

    x = kp.apply_lagrangian_heuristic(x)
    z = kp.evaluate_solution(x)
    print(f"Feasible solution extracted: {x}")
    print(f"Primal bound: {z}")