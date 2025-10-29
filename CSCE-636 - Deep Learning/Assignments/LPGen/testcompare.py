import numpy as np
from scipy.optimize import linprog
import gurobipy as gp
from gurobipy import GRB
import itertools
from tqdm import tqdm
import os

# ==============================================================================
# SOLVER FUNCTION 1: SciPy (HiGHS)
# ==============================================================================

def solve_lp_scipy(args_tuple):
    """
    Solves the linear program using SciPy's linprog.
    """
    G, n, k, a, b, X, psi = args_tuple
    
    all_indices = set(range(n))
    Y = all_indices - X - {a, b}
    m = len(psi)
    x = [0] * n
    x[0] = a; x[m] = b
    x[1:m] = sorted(list(X))
    x[m+1:] = sorted(list(Y))
    tau_inverse = {val: idx for idx, val in enumerate(x)}
    s0 = psi[0]
    
    c = -s0 * G[:, a]
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for j in X:
        s_tau_inv_j = psi[tau_inverse[j]]
        A_ub.append(s_tau_inv_j * G[:, j] - s0 * G[:, a])
        b_ub.append(0)
        A_ub.append(-s_tau_inv_j * G[:, j])
        b_ub.append(-1)
    if b is not None:
        A_eq.append(G[:, b])
        b_eq.append(1)
    for j in Y:
        A_ub.append(G[:, j])
        b_ub.append(1)
        A_ub.append(-G[:, j])
        b_ub.append(1)
    
    A_ub = np.array(A_ub) if A_ub else np.empty((0, k))
    b_ub = np.array(b_ub) if b_ub else np.empty((0,))
    A_eq = np.array(A_eq) if A_eq else np.empty((0, k))
    b_eq = np.array(b_eq) if b_eq else np.empty((0,))

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method='highs')

    if result.success:
        return -result.fun
    elif result.status == 3:
        return np.inf
    elif result.status == 2:
        return 0
    else:
        return 0

# ==============================================================================
# SOLVER FUNCTION 2: Gurobi
# ==============================================================================

def solve_lp_gurobi(args_tuple, gurobi_env):
    """
    Solves the linear program using the high-performance Gurobi solver.
    Accepts a Gurobi environment to suppress license messages.
    """
    G, n, k, a, b, X, psi = args_tuple

    all_indices = set(range(n)); Y = all_indices - X - {a, b}
    m = len(psi); x = [0] * n
    x[0] = a; x[m] = b; x[1:m] = sorted(list(X)); x[m+1:] = sorted(list(Y))
    tau_inverse = {val: idx for idx, val in enumerate(x)}
    s0 = psi[0]

    try:
        model = gp.Model("lp", env=gurobi_env)
        u = model.addVars(k, lb=-GRB.INFINITY, name="u")
        model.setObjective(gp.LinExpr([(s0 * G[i, a]) for i in range(k)], [u[i] for i in range(k)]), GRB.MAXIMIZE)

        for j in X:
            s_tau_inv_j = psi[tau_inverse[j]]
            model.addConstr(gp.LinExpr([(s_tau_inv_j * G[i, j] - s0 * G[i, a]) for i in range(k)], [u[i] for i in range(k)]) <= 0)
            model.addConstr(gp.LinExpr([(-s_tau_inv_j * G[i, j]) for i in range(k)], [u[i] for i in range(k)]) <= -1)
        model.addConstr(gp.LinExpr([(G[i, b]) for i in range(k)], [u[i] for i in range(k)]) == 1)
        for j in Y:
            model.addConstr(gp.LinExpr([(G[i, j]) for i in range(k)], [u[i] for i in range(k)]) <= 1)
            model.addConstr(gp.LinExpr([(-G[i, j]) for i in range(k)], [u[i] for i in range(k)]) <= 1)
        
        model.setParam('OutputFlag', 0)
        model.optimize()

        if model.status == GRB.OPTIMAL: return model.ObjVal
        elif model.status == GRB.UNBOUNDED: return np.inf
        elif model.status == GRB.INFEASIBLE: return 0
        else: return 0
    except gp.GurobiError:
        return 0


# ==============================================================================
# MAIN COMPARISON LOGIC
# ==============================================================================

if __name__ == '__main__':
    # --- Configuration ---
    # Choose a single (n, k, m) pair to test.
    # A pair with a smaller number of LPs is better for a quick test.
    N_test, K_test, M_test = 9, 6, 2

    print("===================================================")
    print("      SciPy vs. Gurobi Solver Comparison")
    print("===================================================\n")
    print(f"Test Parameters: n={N_test}, k={K_test}, m={M_test}")

    # --- Step 1: Generate a single, consistent test case ---
    print("\n[1] Generating a random G matrix to use for both solvers...")
    P_test = np.random.uniform(-100, 100, size=(K_test, N_test - K_test))
    I_test = np.identity(K_test)
    G_test = np.hstack([I_test, P_test])
    print("   Done.")

    # --- Step 2: Pre-generate all LP problem arguments ---
    print("\n[2] Generating all possible (a, b, X, psi) tuples...")
    all_lp_args = []
    for a, b in itertools.permutations(range(N_test), 2):
        remaining_indices = list(range(N_test))
        remaining_indices.remove(a)
        remaining_indices.remove(b)
        for X_tuple in itertools.combinations(remaining_indices, M_test - 1):
            for psi in itertools.product([-1, 1], repeat=M_test):
                all_lp_args.append((G_test, N_test, K_test, a, b, set(X_tuple), psi))
    print(f"   Generated {len(all_lp_args)} unique LP problems to solve.")

    # --- Step 3: Run the SciPy solver ---
    print("\n[3] Running SciPy solver on all LP problems...")
    scipy_results = []
    for args in tqdm(all_lp_args, desc="SciPy"):
        scipy_results.append(solve_lp_scipy(args))
    print("   SciPy run complete.")

    # --- Step 4: Run the Gurobi solver ---
    print("\n[4] Running Gurobi solver on all LP problems...")
    gurobi_results = []
    # Use a Gurobi environment to suppress the "academic license" message on every call
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        for args in tqdm(all_lp_args, desc="Gurobi"):
            gurobi_results.append(solve_lp_gurobi(args, env))
    print("   Gurobi run complete.")

    # --- Step 5: Compare the results ---
    print("\n[5] Comparing results from both solvers...")
    consistent_count = 0
    inconsistent_count = 0

    for i, (res_scipy, res_gurobi) in enumerate(zip(scipy_results, gurobi_results)):
        is_consistent = False
        # Case 1: Both are infinite (unbounded)
        if res_scipy == np.inf and res_gurobi == np.inf:
            is_consistent = True
        # Case 2: Both are finite numbers and are very close
        elif res_scipy != np.inf and res_gurobi != np.inf:
            if np.isclose(res_scipy, res_gurobi, atol=1e-6):
                is_consistent = True
        
        if is_consistent:
            consistent_count += 1
        else:
            inconsistent_count += 1
            print("-" * 20)
            print(f"  INCONSISTENCY FOUND at index {i}!")
            print(f"  Inputs (a,b,X,psi): {all_lp_args[i][3:]}")
            print(f"  SciPy Result:  {res_scipy}")
            print(f"  Gurobi Result: {res_gurobi}")
            print("-" * 20)

    print("\n--- COMPARISON SUMMARY ---")
    print(f"Total LPs solved:      {len(all_lp_args)}")
    print(f"Consistent results:    {consistent_count}")
    print(f"Inconsistent results:  {inconsistent_count}")

    if inconsistent_count == 0:
        print("\n✅ All LP results are consistent between SciPy and Gurobi.")
    else:
        print("\n❌ Inconsistencies were found. Please review the output above.")

    # --- Final check on the m-height ---
    m_height_scipy = np.max(scipy_results)
    m_height_gurobi = np.max(gurobi_results)

    print("\n--- FINAL M-HEIGHT CALCULATION ---")
    print(f"m-height calculated by SciPy:  {m_height_scipy}")
    print(f"m-height calculated by Gurobi: {m_height_gurobi}")

    if np.isclose(m_height_scipy, m_height_gurobi, atol=1e-6):
        print("\n✅ The final m-height values are consistent!")
    else:
        print("\n❌ WARNING: The final m-height values DO NOT MATCH.")
    
    print("\n===================================================")
    print("               Comparison Complete")
    print("===================================================")