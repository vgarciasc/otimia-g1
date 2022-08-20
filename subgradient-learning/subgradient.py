import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
import rich

from knapsack_problem import BinaryKnapsackProblem

from rich.console import Console
console = Console()

def solve_01kp_subgradient(bkp, max_iterations=100, verbose=True):
    rho_start = 2
    rho_min = 0.001
    theta = 1
    epsilon = 0.1

    x_lower = None
    z_lower = - math.inf
    z_upper = + math.inf

    u = 0
    rho = rho_start
    is_solved = False
    history = []

    for k in range(max_iterations):
        # Obtain upper bound by solving the lagrangian subproblem
        x_u, z_u = bkp.solve_lagrangian_subproblem(u)

        # Obtain lower bound by applying an heuristic to the subproblem sol.
        x_k, z_k = bkp.apply_lagrangian_heuristic_densityprob(x_u)

        #x_k is a feasible solution in the original problem
        #x_u is a solution for the lagrangian relaxation

        if z_u < z_upper:
            z_upper = z_u

        if z_k > z_lower:
            x_lower = x_k
            z_lower = z_k
            improve = 0
        else:
            improve += 1

            if improve >= max_iterations / 20:
                rho /= 2
                improve = 0

                if rho < rho_min:
                    console.log(f"\n[red]Stopping after {k} iterations:"
                        + "rho < rho_min => {rho} < {rho_min}!")
                    break
        
        # Update the lagrangian multipliers using the subgradient
        G_k = theta * (C - np.sum([x_u[i] * w[i] for i in range(bkp.N)]))
        T = rho * (z_upper - z_k) / G_k**2
        u = np.max([u - (1 + epsilon) * T * G_k, 0])

        # Simplified update rule
        # u = np.max([u - 0.5**k * G_k, 0])
    
        # Logging iteration
        history.append((z_u, z_k, z_upper, z_lower))

        if verbose:
            console.rule(f"Iteration {k}")
            console.log(f"[cyan]    z_k: {z_k}")
            console.log(f"[cyan]    z_u: {z_u}")
            console.log(f"[green]z_lower: {z_lower}")
            console.log(f"[green]z_upper: {z_upper}")
            console.log(f"[blue]opt gap: {'{:.3f}'.format((z_upper - z_lower)/z_upper)}")
            console.log(f"[yellow]    G_k: {G_k}")
            console.log(f"[yellow]    T_k: {T}")
            console.log(f"[yellow]    u_k: {u}")
        
        # Checking for convergence
        if z_upper - z_lower <= 1:
            is_solved = True
            console.log(f"\n[red]Stopping after {k} iterations: solved!")
            break
    
    return is_solved, x_lower, z_lower, z_upper, history

if __name__ == "__main__":
    # Classroom instance
    # w = [3, 1, 4]
    # p = [10, 4, 14]
    # C = 4

    # Instance #1
    # w = [4, 2, 5, 4, 5, 1, 3, 5]
    # p = [10, 5, 18, 12, 15, 1, 2, 8]
    # C = 15

    # Instance #2
    w = [995, 485, 326, 248, 421, 322, 795, 43, 845, 955, 252, 9, 901, 122, 94, 738, 574, 715, 882, 367, 984, 299, 433, 682, 72, 874, 138, 856, 145, 995, 529, 199, 277, 97, 719, 242, 107, 122, 70, 98, 600, 645, 267, 972, 895, 213, 748, 487, 923, 29, 674, 540, 554, 467, 46, 710, 553, 191, 724, 730, 988, 90, 340, 549, 196, 865, 678, 570, 936, 722, 651, 123, 431, 508, 585, 853, 642, 992, 725, 286, 812, 859, 663, 88, 179, 187, 619, 261, 846, 192, 261, 514, 886, 530, 849, 294, 799, 391, 330, 298, 790]
    p = [100, 94, 506, 416, 992, 649, 237, 457, 815, 446, 422, 791, 359, 667, 598, 7, 544, 334, 766, 994, 893, 633, 131, 428, 700, 617, 874, 720, 419, 794, 196, 997, 116, 908, 539, 707, 569, 537, 931, 726, 487, 772, 513, 81, 943, 58, 303, 764, 536, 724, 789, 479, 142, 339, 641, 196, 494, 66, 824, 208, 711, 800, 314, 289, 401, 466, 689, 833, 225, 244, 849, 113, 379, 361, 65, 486, 686, 286, 889, 24, 491, 891, 90, 181, 214, 17, 472, 418, 419, 356, 682, 306, 201, 385, 952, 500, 194, 737, 324, 992, 224]
    C = 1000

    w = np.array(w)
    w, C = w / C, 1
    N = len(w)

    bkp = BinaryKnapsackProblem(w, p, C)
    print(bkp)

    results = solve_01kp_subgradient(bkp, max_iterations=100, verbose=True)
    is_solved, x_lower, z_lower, z_upper, history = results
    z_us, z_ks, z_uppers, z_lowers = zip(*history)
    iterations = len(z_us)

    console.rule("[bold red]Results")
    console.log(f"[blue]    iterations: {iterations}")
    console.log(f"[cyan]optimality gap: {'{:.3f}'.format((z_upper - z_lower)/z_upper)}")
    console.log(f"[cyan]   z_u (upper): {z_upper}")
    console.log(f"[cyan]   z_k (lower): {z_lower}")
    console.log(f"\nfinal solution value: {bkp.evaluate_solution(x_lower)}")
    console.log(f"\nfinal solution:")
    console.log(x_lower)

    plt.plot(range(iterations), z_us, '-r', label='$z(u)$')
    plt.plot(range(iterations), z_uppers, '--r')
    plt.plot(range(iterations), z_ks, '-b', label='$z_k$')
    plt.plot(range(iterations), z_lowers, '--b')
    plt.xlabel("Iteration")
    plt.title(f"0-1 Knapsack Problem\nIterations: {iterations}, Opt. Gap: {'{:.3f}'.format((z_upper - z_lower)/z_upper)}, Best Solution: {z_lower}")
    plt.legend()
    plt.show()