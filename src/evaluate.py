def evaluate_agent(env, agent, num_episodes=20, record_trajectories=True):
    """
    Enhanced evaluation function that captures multiple performance metrics relevant to clinical trials
    
    Parameters:
    -----------
    env : EnhancedTrialEnv
        Environment to evaluate in
    agent : SACAgent
        Agent to evaluate
    num_episodes : int
        Number of episodes to evaluate over (increased from 10 to 20 for lower variance)
    record_trajectories : bool
        Whether to record detailed trajectory information
        
    Returns:
    --------
    dict
        Dictionary containing multiple evaluation metrics
    """
    # Containers for metrics
    success_proportions = []
    total_rewards = []
    info_gains = []
    exp_arm_allocations = []
    ctrl_arm_allocations = []
    final_tvds = []
    prob_exp_better_final = []
    trajectories = [] if record_trajectories else None
    
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        # For tracking allocation trajectory
        trajectory = [] if record_trajectories else None
        
        # Calculate initial TVD for info gain measurement
        initial_tvd = compute_tvd(env.alpha_exp, env.beta_exp, env.alpha_ctrl, env.beta_ctrl)
        
        while not done:
            # Select action (deterministic policy for evaluation)
            action = agent.select_action(state, evaluate=True)
            
            # Track action/allocation
            alloc_exp = int(round(action * env.N))
            alloc_ctrl = env.N - alloc_exp
            
            if record_trajectories:
                # Record the state, action, and current beliefs
                exp_rate = env.alpha_exp / (env.alpha_exp + env.beta_exp)
                ctrl_rate = env.alpha_ctrl / (env.alpha_ctrl + env.beta_ctrl)
                prob_better = compute_prob_better(env.alpha_exp, env.beta_exp, 
                                                  env.alpha_ctrl, env.beta_ctrl)
                
                trajectory.append({
                    'state': state.copy(),
                    'action': action,
                    'allocation_exp': alloc_exp,
                    'allocation_ctrl': alloc_ctrl,
                    'exp_rate': exp_rate,
                    'ctrl_rate': ctrl_rate,
                    'prob_exp_better': prob_better,
                    'alpha_exp': env.alpha_exp,
                    'beta_exp': env.beta_exp,
                    'alpha_ctrl': env.alpha_ctrl,
                    'beta_ctrl': env.beta_ctrl,
                    'period': env.current_period
                })
            
            # Take step
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
                
        # Calculate final metrics
        success_proportion = env.total_successes / (env.N * env.T)
        success_proportions.append(success_proportion)
        total_rewards.append(total_reward)
        
        # Calculate information gain (change in TVD)
        final_tvd = compute_tvd(env.alpha_exp, env.beta_exp, env.alpha_ctrl, env.beta_ctrl)
        info_gain = final_tvd - initial_tvd
        info_gains.append(info_gain)
        final_tvds.append(final_tvd)
        
        # Track arm allocations
        exp_arm_allocations.append(env.allocation_history_exp / (env.allocation_history_exp + env.allocation_history_ctrl))
        ctrl_arm_allocations.append(env.allocation_history_ctrl / (env.allocation_history_exp + env.allocation_history_ctrl))
        
        # Track final probability that experimental treatment is better
        prob_exp_better = compute_prob_better(env.alpha_exp, env.beta_exp, env.alpha_ctrl, env.beta_ctrl)
        prob_exp_better_final.append(prob_exp_better)
        
        if record_trajectories:
            trajectories.append(trajectory)
    
    # Calculate means and standard errors
    mean_success = np.mean(success_proportions)
    se_success = np.std(success_proportions) / np.sqrt(num_episodes)
    
    mean_exp_allocation = np.mean(exp_arm_allocations)
    se_exp_allocation = np.std(exp_arm_allocations) / np.sqrt(num_episodes)
    
    mean_info_gain = np.mean(info_gains)
    se_info_gain = np.std(info_gains) / np.sqrt(num_episodes)
    
    mean_prob_better = np.mean(prob_exp_better_final)
    se_prob_better = np.std(prob_exp_better_final) / np.sqrt(num_episodes)
    
    # Calculate reward decomposition (approximate)
    mean_reward = np.mean(total_rewards)
    approx_immediate_reward = mean_success * env.N * env.T
    approx_info_gain_reward = mean_info_gain * env.lambda_tvd * env.T
    approx_explore_bonus = mean_reward - approx_immediate_reward - approx_info_gain_reward
    
    results = {
        # Primary clinical metric
        'success_proportion': (mean_success, se_success),
        
        # Allocation metrics
        'exp_allocation': (mean_exp_allocation, se_exp_allocation),
        'ctrl_allocation': (1 - mean_exp_allocation, se_exp_allocation),
        
        # Knowledge-related metrics
        'info_gain': (mean_info_gain, se_info_gain),
        'final_tvd': np.mean(final_tvds),
        'prob_exp_better': (mean_prob_better, se_prob_better),
        
        # Reward decomposition
        'total_reward': mean_reward,
        'reward_components': {
            'immediate_reward': approx_immediate_reward,
            'info_gain_reward': approx_info_gain_reward,
            'exploration_bonus': approx_explore_bonus
        },
        
        # Raw data for further analysis
        'success_proportions': success_proportions,
        'exp_arm_allocations': exp_arm_allocations,
        'info_gains': info_gains,
        'trajectories': trajectories
    }
    
    return results

# Heuristic Policy Functions we test SAC against (benchmarks essentially)
def heuristic_fixed(env):
    return 0.5

def heuristic_greedy(env):
    p_exp = env.alpha_exp / (env.alpha_exp + env.beta_exp)
    p_ctrl = env.alpha_ctrl / (env.alpha_ctrl + env.beta_ctrl)
    return 1.0 if p_exp > p_ctrl else 0.0

def heuristic_PL(env):
    a1, b1 = env.alpha_exp, env.beta_exp
    a2, b2 = env.alpha_ctrl, env.beta_ctrl
    prob_exp_best = compute_prob_better(a1, b1, a2, b2)
    return 1.0 if prob_exp_best > 0.5 else 0.0

# Ideal policy (oracle)
def ideal_policy(env):
    if env.p_true_exp is not None and env.p_true_ctrl is not None:
        return 1.0 if env.p_true_exp > env.p_true_ctrl else 0.0
    else:
        p_exp = env.alpha_exp / (env.alpha_exp + env.beta_exp)
        p_ctrl = env.alpha_ctrl / (env.alpha_ctrl + env.beta_ctrl)
        return 1.0 if p_exp > p_ctrl else 0.0

def run_oracle_benchmark(N, T, p_true_exp, p_true_ctrl, seed=None, safety_factor=1.5):
    """
    Generate all potential outcomes and find the optimal allocation with perfect knowledge
    
    Parameters:
    -----------
    N : int
        Number of patients per cohort
    T : int
        Number of time periods
    p_true_exp : float
        True success probability for experimental treatment
    p_true_ctrl : float
        True success probability for control treatment
    seed : int or None
        Random seed for reproducibility
    safety_factor : float
        Factor to multiply N*T by to ensure sufficient outcomes (default: 1.5)
        
    Returns:
    --------
    best_allocation : int
        Optimal number of patients to allocate to experimental treatment
    success_proportion : float
        Success proportion achieved with optimal allocation
    all_outcomes_exp : list
        Pre-generated outcomes for experimental treatment
    all_outcomes_ctrl : list
        Pre-generated outcomes for control treatment
    """
    if seed is not None:
        rng = np.random.RandomState(generate_benchmark_seed(seed))
    else:
        rng = np.random.RandomState()
    
    # Calculate total number of potential patients
    total_patients = N * T
    
    # Generate sufficient outcomes with safety margin
    # This ensures we have enough outcomes for all possible allocations
    # and provides a buffer for adaptive or varying allocation strategies
    num_outcomes = int(total_patients * safety_factor)
    
    # Generate all potential outcomes for both treatments
    all_outcomes_exp = []
    all_outcomes_ctrl = []
    
    # Generate outcomes
    all_outcomes_exp = [1 if rng.random() < p_true_exp else 0 for _ in range(num_outcomes)]
    all_outcomes_ctrl = [1 if rng.random() < p_true_ctrl else 0 for _ in range(num_outcomes)]
    
    # Verify we have enough outcomes
    if len(all_outcomes_exp) < total_patients or len(all_outcomes_ctrl) < total_patients:
        warnings.warn(f"Generated only {len(all_outcomes_exp)} outcomes when {total_patients} might be needed. "
                     f"Consider increasing safety_factor above {safety_factor}.")
    
    # Find optimal allocation using perfect knowledge
    best_allocation = None
    max_successes = -1
    
    # Try all possible allocations of n patients to experimental treatment
    for n_exp in range(total_patients + 1):
        n_ctrl = total_patients - n_exp
        
        # Calculate successes for this allocation
        exp_successes = sum(all_outcomes_exp[:n_exp])
        ctrl_successes = sum(all_outcomes_ctrl[:n_ctrl])
        total_successes = exp_successes + ctrl_successes
        
        if total_successes > max_successes:
            max_successes = total_successes
            best_allocation = n_exp
    
    # Calculate the success proportion
    success_proportion = max_successes / total_patients
    
    # Create metadata for tracking
    metadata = {
        'N': N,
        'T': T,
        'p_true_exp': p_true_exp,
        'p_true_ctrl': p_true_ctrl,
        'seed': seed,
        'best_allocation': best_allocation,
        'best_allocation_ratio': best_allocation / total_patients,
        'max_successes': max_successes,
        'success_proportion': success_proportion,
        'outcomes_generated': len(all_outcomes_exp)
    }
    
    return best_allocation, success_proportion, all_outcomes_exp, all_outcomes_ctrl, metadata


def calculate_statistical_significance(success_proportions, policy_names):
    """
    Calculate statistical significance of differences between policies
    
    Parameters:
    -----------
    success_proportions : dict
        Dictionary mapping policy names to lists of success proportions
    policy_names : list
        List of policy names
        
    Returns:
    --------
    significance : dict
        Dictionary containing statistical significance results
    """
    
    significance = {}
    
    for i, name1 in enumerate(policy_names):
        for j, name2 in enumerate(policy_names):
            if j <= i:  # Only compare each pair once
                continue
                
            # Get data
            data1 = success_proportions[name1]
            data2 = success_proportions[name2]
            
            # Perform t-test
            t_stat, p_val = stats.ttest_ind(data1, data2)
            
            # Calculate effect size (Cohen's d)
            effect_size = (np.mean(data1) - np.mean(data2)) / np.sqrt(
                (np.std(data1, ddof=1)**2 + np.std(data2, ddof=1)**2) / 2
            )
            
            # Store results
            key = f"{name1}_vs_{name2}"
            significance[key] = {
                't_statistic': t_stat,
                'p_value': p_val,
                'effect_size': effect_size,
                'significant': p_val < 0.05
            }
    
    return significance


def generate_benchmark_seed(base_seed, salt=""):
    """
    Generate a reproducible but well-distributed seed using a hash function
    
    Parameters:
    -----------
    base_seed : int
        Base seed value
    salt : str
        Additional string to mix with the seed
        
    Returns:
    --------
    derived_seed : int
        New seed value derived from base_seed and salt
    """
    seed_str = f"{base_seed}_{salt}"
    
    hash_obj = hashlib.sha256(seed_str.encode())
    hash_hex = hash_obj.hexdigest()
    
    return int(hash_hex[:8], 16)
