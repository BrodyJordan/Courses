import numpy as np
from scipy.optimize import linprog
import itertools
from tqdm import tqdm
import pickle
import os
import multiprocessing

# ==============================================================================
# CORE SCIENTIFIC FUNCTIONS
# ==============================================================================

def solve_lp(args_tuple):
    """
    Solves the linear program LPa,b,X,ψ.
    Accepts a single tuple of arguments for easier mapping.
    """
    G, n, k, a, b, X, psi = args_tuple
    
    all_indices = set(range(n))
    Y = all_indices - X - {a, b}
    m = len(psi)
    x = [0] * n
    x[0] = a
    x[m] = b
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

def parallel_calculate_m_height(G, n, k, m, pool):
    """
    A parallelized version of calculate_m_height that uses an existing pool.
    """
    all_lp_args = []
    for a, b in itertools.permutations(range(n), 2):
        remaining_indices = list(range(n))
        remaining_indices.remove(a)
        remaining_indices.remove(b)
        for X_tuple in itertools.combinations(remaining_indices, m - 1):
            for psi in itertools.product([-1, 1], repeat=m):
                all_lp_args.append((G, n, k, a, b, set(X_tuple), psi))

    results = pool.map(solve_lp, all_lp_args)
    
    max_z = np.max(results)
    return max_z if max_z > 0 and np.isfinite(max_z) else 1.0


def generate_dataset(n, k, m, num_samples, pool):
    """
    Generates a dataset using a serial loop, where each iteration calls a parallel calculation.
    """
    training_data = [] 
    training_targets = []

    for _ in tqdm(range(num_samples), desc=f"Generating samples for ({n},{k},{m})"):
        P = np.random.uniform(-100, 100, size=(k, n - k))
        I = np.identity(k)
        G = np.hstack([I, P])
        
        m_height = parallel_calculate_m_height(G, n, k, m, pool)
        
        if m_height == np.inf:
            continue
            
        training_data.append([n, k, m, P])
        training_targets.append(m_height)

    return training_data, training_targets

# ==============================================================================
# MAIN SCRIPT LOGIC
# ==============================================================================

# The 'if __name__ == '__main__':' guard is ESSENTIAL for cross-platform
# compatibility, especially on Windows.
if __name__ == '__main__':
    
    # --- REMOVED THIS LINE ---
    # multiprocessing.set_start_method('fork', force=True) 
    # This allows Python to choose the correct method ('spawn' on Windows).

    # --- Configuration ---
    TARGETS = [
        # {'n': 9, 'k': 4, 'm': 2, 'current': 2677, 'goal': 10000},
        {'n': 9, 'k': 4, 'm': 3, 'current': 2588, 'goal': 10000},
        {'n': 9, 'k': 4, 'm': 4, 'current': 2376, 'goal': 10000},
        {'n': 9, 'k': 4, 'm': 5, 'current': 1494, 'goal': 10000},
        {'n': 9, 'k': 5, 'm': 2, 'current': 3579, 'goal': 10000},
        {'n': 9, 'k': 5, 'm': 3, 'current': 3286, 'goal': 10000},
        {'n': 9, 'k': 5, 'm': 4, 'current': 2002, 'goal': 10000},
        {'n': 9, 'k': 6, 'm': 2, 'current': 9336, 'goal': 10000},
        {'n': 9, 'k': 6, 'm': 3, 'current': 4749, 'goal': 10000},
    ]


    print("===================================================")
    print("Starting Large-Scale Data Generation Campaign")
    print(f"Using {multiprocessing.cpu_count()} CPU cores for parallel processing.")
    print("===================================================\n")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as main_pool:
        for job in TARGETS:
            n, k, m = job['n'], job['k'], job['m']
            
            output_dir = "training_data"
            os.makedirs(output_dir, exist_ok=True)
            filename_data = os.path.join(output_dir, f"data_{n}_{k}_{m}.pkl")
            filename_targets = os.path.join(output_dir, f"targets_{n}_{k}_{m}.pkl")

            try:
                with open(filename_data, "rb") as f:
                    current_samples = len(pickle.load(f))
            except FileNotFoundError:
                current_samples = 0
            
            num_to_generate = job['goal'] - job['current']

            print(f"--- Processing Target: (n={n}, k={k}, m={m}) ---")
            
            if num_to_generate <= 0:
                print(f"Goal of {job['goal']} samples already met. Currently have {current_samples}. Skipping.\n")
                continue

            print(f"Current samples: {current_samples}. Goal: {job['goal']}.")
            print(f"Need to generate {num_to_generate} new samples.")

            existing_data = []
            existing_targets = []
            if current_samples > 0:
                with open(filename_data, "rb") as f:
                    existing_data = pickle.load(f)
                with open(filename_targets, "rb") as f:
                    existing_targets = pickle.load(f)
                print(f"Loaded {len(existing_data)} existing samples.")

            new_data, new_targets = generate_dataset(n=n, k=k, m=m, num_samples=num_to_generate, pool=main_pool)
            
            combined_data = existing_data + new_data
            combined_targets = existing_targets + new_targets

            print(f"\nGenerated {len(new_data)} new valid samples.")
            print(f"Total samples for ({n},{k},{m}) is now {len(combined_data)}.")

            with open(filename_data, "wb") as f:
                pickle.dump(combined_data, f)
            with open(filename_targets, "wb") as f:
                pickle.dump(combined_targets, f)
            
            print(f"Successfully saved combined data to {filename_data}\n")

    print("===================================================")
    print("Data Generation Campaign Finished!")
    print("===================================================")