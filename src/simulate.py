def generate_seed(seed_start, index, salt=""):
    """
    Generate a unique and well-distributed seed using a hash function
    
    This avoids the sequential correlation that can occur with simple seed_start + i
    The hash function creates more entropy in the seed distribution.
    """
    # Combine seed_start, index, and optional salt (e.g., function name)
    seed_str = f"{seed_start}_{index}_{salt}"
    
    # Use hashlib to get a hash, then convert to an integer
    hash_obj = hashlib.sha256(seed_str.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Convert first 8 chars of hex to an integer
    return int(hash_hex[:8], 16)

def simulation_worker(args):
    policy_fn, env_params, index, seed_start, track_metrics, agent, policy_name = args
    return run_policy_simulation(policy_fn, env_params, index, seed_start, track_metrics, agent, policy_name)

def simulate_trial_base(policy_fn, env_params, num_sim=100, seed_start=0, 
                        track_metrics=True, agent=None, policy_name="default",
                        parallel=False, num_processes=None):
    """
    Base simulation function that works with any policy
    
    Parameters:
    -----------
    policy_fn : function
        Function that takes the environment and returns an action
    env_params : dict
        Parameters for the environment
    num_sim : int
        Number of simulations to run
    seed_start : int
        Starting seed for random number generation
    track_metrics : bool
        Whether to track additional metrics beyond success proportion
    agent : object
        Agent object (used only when policy_fn requires it)
    policy_name : str
        Name of the policy (for seed generation)
    parallel : bool
        Whether to run simulations in parallel
    num_processes : int or None
        Number of processes for parallel execution, if None uses CPU count
        
    Returns:
    --------
    dict
        Dictionary containing simulation results
    """
    if parallel:
        # Setup parallel processing
        if num_processes is None:
            num_processes = max(1, mp.cpu_count() - 1)  # Leave 1 CPU free
            
        # Create argument list for parallel processing
        args_list = [(policy_fn, env_params, i, seed_start, track_metrics, agent, policy_name) 
                     for i in range(num_sim)]

        # Run simulations in parallel
        try:
            with mp.Pool(processes=num_processes) as pool:
                results = list(tqdm(pool.imap(simulation_worker, args_list), 
                                   total=num_sim, desc=f"Running {policy_name} simulations"))
                
        except Exception as e:
            print(f"Error in parallel execution: {e}")
            print("Falling back to sequential execution...")
            results = [run_policy_simulation(policy_fn, env_params, i, seed_start, 
                                          track_metrics, agent, policy_name) 
                       for i in tqdm(range(num_sim), desc=f"Running {policy_name} simulations")]
    else:
        # Run simulations sequentially
        results = [run_policy_simulation(policy_fn, env_params, i, seed_start, 
                                      track_metrics, agent, policy_name) 
                   for i in tqdm(range(num_sim), desc=f"Running {policy_name} simulations")]
    
    success_proportions = [result['success_proportion'] for result in results]
    
    mean_prop = np.mean(success_proportions)
    std_err = np.std(success_proportions) / np.sqrt(num_sim)
    
    ci_95 = stats.t.interval(0.95, len(success_proportions)-1, 
                             loc=mean_prop, 
                             scale=std_err)
                             
    aggregated_results = {
        'success_proportion': (mean_prop, std_err),
        'confidence_interval_95': ci_95,
        'success_proportions': success_proportions,
    }
    
    if track_metrics:
        aggregated_results.update(aggregate_advanced_metrics(results))
    
    return aggregated_results

def run_policy_simulation(policy_fn, env_params, index, seed_start, track_metrics, agent, policy_name):
    """Run a single simulation and return its results"""
    env_params_copy = env_params.copy()
    
    # Generate a unique, non-sequential seed for this simulation
    sim_seed = generate_seed(seed_start, index, salt=policy_name)
    env_params_copy['seed'] = sim_seed
    
    # Create environment
    env = EnhancedTrialEnv(**env_params_copy)
    
    # Initialize metrics
    metrics = {
        'rewards': [],
        'allocations_exp': [],
        'allocations_ctrl': [],
        'successes_exp': [],
        'successes_ctrl': [],
        'tvd_values': []
    }

    # Initialize trajectory list if tracking is enabled
    trajectory = [] if track_metrics else None
    
    state = env.reset()
    
    # Track initial TVD
    if track_metrics:
        initial_tvd = compute_tvd(env.alpha_exp, env.beta_exp, env.alpha_ctrl, env.beta_ctrl)
        metrics['tvd_values'].append(initial_tvd)

    step_idx = 0
    while True:
        # Get action based on the provided policy function
        if policy_name == "sac" and agent is not None:
            action = policy_fn(state, evaluate=True)
        else:
            action = policy_fn(env)
            
        # Convert action to allocations
        alloc_exp = int(round(action * env.N))
        alloc_ctrl = env.N - alloc_exp
        
        # Take step
        next_state, reward, done = env.step(action)
        
        # Track metrics if requested
        if track_metrics:
            metrics['rewards'].append(reward)
            metrics['allocations_exp'].append(alloc_exp)
            metrics['allocations_ctrl'].append(alloc_ctrl)
            
            # Calculate successes for this step
            if hasattr(env, 'alpha_exp_prev') and hasattr(env, 'alpha_ctrl_prev'):
                successes_exp = env.alpha_exp - env.alpha_exp_prev
                successes_ctrl = env.alpha_ctrl - env.alpha_ctrl_prev
            else:
                # Rough approximation
                p_exp = env.alpha_exp / (env.alpha_exp + env.beta_exp)
                p_ctrl = env.alpha_ctrl / (env.alpha_ctrl + env.beta_ctrl)
                successes_exp = alloc_exp * p_exp
                successes_ctrl = alloc_ctrl * p_ctrl
            
            metrics['successes_exp'].append(successes_exp)
            metrics['successes_ctrl'].append(successes_ctrl)
            
            # Track TVD
            current_tvd = compute_tvd(env.alpha_exp, env.beta_exp, env.alpha_ctrl, env.beta_ctrl)
            metrics['tvd_values'].append(current_tvd)

            exp_rate = env.alpha_exp / (env.alpha_exp + env.beta_exp)
            ctrl_rate = env.alpha_ctrl / (env.alpha_ctrl + env.beta_ctrl)
            
            # Append step information to the trajectory
            trajectory.append({
                'period': step_idx,
                'allocation_exp': alloc_exp,
                'allocation_ctrl': alloc_ctrl,
                'exp_rate': exp_rate,
                'ctrl_rate': ctrl_rate,
                'alpha_exp': env.alpha_exp,
                'beta_exp': env.beta_exp,
                'alpha_ctrl': env.alpha_ctrl,
                'beta_ctrl': env.beta_ctrl
            })
        
        # Update state and check if done
        state = next_state
        step_idx += 1
        if done:
            break
    
    # Calculate success proportion
    success_proportion = env.total_successes / (env.N * env.T)
    
    # Prepare results
    result = {'success_proportion': success_proportion}
    
    # Add additional metrics if requested
    if track_metrics:
        # Calculate allocation percentages
        total_exp = sum(metrics['allocations_exp'])
        total_ctrl = sum(metrics['allocations_ctrl'])
        total_patients = total_exp + total_ctrl
        
        # Calculate final TVD - initial TVD for information gain
        info_gain = metrics['tvd_values'][-1] - metrics['tvd_values'][0]
        
        # Calculate reward components
        total_reward = sum(metrics['rewards'])
        success_reward = env.total_successes  # Immediate reward component
        
        # Calculate standard deviation of allocations over time
        if len(metrics['allocations_exp']) > 1:
            allocation_ratios = [a / env.N for a in metrics['allocations_exp']]
            allocation_variability = np.std(allocation_ratios)
        else:
            allocation_variability = 0
            
        # Calculate final probability that experimental treatment is better
        prob_exp_better = compute_prob_better(env.alpha_exp, env.beta_exp, 
                                             env.alpha_ctrl, env.beta_ctrl)
        
        # Extended metrics
        result.update({
            'exp_allocation_pct': total_exp / total_patients,
            'ctrl_allocation_pct': total_ctrl / total_patients,
            'total_reward': total_reward,
            'immediate_reward': success_reward,
            'auxiliary_reward': total_reward - success_reward,
            'info_gain': info_gain,
            'allocation_variability': allocation_variability,
            'prob_exp_better': prob_exp_better,
            'final_alpha_exp': env.alpha_exp,
            'final_beta_exp': env.beta_exp,
            'final_alpha_ctrl': env.alpha_ctrl,
            'final_beta_ctrl': env.beta_ctrl,
            'raw_metrics': metrics
        })
        
        result['trajectory'] = trajectory
    
    return result

def aggregate_advanced_metrics(results):
    """Aggregate advanced metrics from multiple simulation results"""
    # Extract metrics
    exp_allocation_pct = [r['exp_allocation_pct'] for r in results]
    total_rewards = [r['total_reward'] for r in results]
    immediate_rewards = [r['immediate_reward'] for r in results]
    auxiliary_rewards = [r['auxiliary_reward'] for r in results]
    info_gains = [r['info_gain'] for r in results]
    allocation_variability = [r['allocation_variability'] for r in results]
    prob_exp_better = [r['prob_exp_better'] for r in results]
    
    # Calculate means and standard errors
    n = len(results)
    
    aggregated = {
        'exp_allocation': (np.mean(exp_allocation_pct), np.std(exp_allocation_pct) / np.sqrt(n)),
        'ctrl_allocation': (1 - np.mean(exp_allocation_pct), np.std(exp_allocation_pct) / np.sqrt(n)),
        'total_reward': (np.mean(total_rewards), np.std(total_rewards) / np.sqrt(n)),
        'immediate_reward': (np.mean(immediate_rewards), np.std(immediate_rewards) / np.sqrt(n)),
        'auxiliary_reward': (np.mean(auxiliary_rewards), np.std(auxiliary_rewards) / np.sqrt(n)),
        'info_gain': (np.mean(info_gains), np.std(info_gains) / np.sqrt(n)),
        'allocation_variability': (np.mean(allocation_variability), np.std(allocation_variability) / np.sqrt(n)),
        'prob_exp_better': (np.mean(prob_exp_better), np.std(prob_exp_better) / np.sqrt(n)),
        
        # Calculate reward proportion (what % comes from immediate vs auxiliary)
        'immediate_reward_pct': np.mean(immediate_rewards) / np.mean(total_rewards) * 100,
        'auxiliary_reward_pct': np.mean(auxiliary_rewards) / np.mean(total_rewards) * 100,
    }

    trajectories = [r.get('trajectory') for r in results if r.get('trajectory') is not None]
    if trajectories:
        aggregated['trajectories'] = trajectories
    
    return aggregated

# Wrapper functions for specific policy types
def simulate_trial_ideal(env_params, num_sim=100, seed_start=0, track_metrics=True, parallel=False):
    """Simulate trial with ideal policy"""
    return simulate_trial_base(ideal_policy, env_params, num_sim, seed_start, 
                             track_metrics, policy_name="ideal", parallel=parallel)

def simulate_trial_heuristic(env_params, heuristic_fn, num_sim=100, seed_start=0, 
                            track_metrics=True, parallel=False):
    """Simulate trial with a heuristic policy"""
    return simulate_trial_base(heuristic_fn, env_params, num_sim, seed_start, 
                             track_metrics, policy_name=heuristic_fn.__name__, parallel=parallel)

def sac_select_action(state, agent, evaluate=True):
    """Wrapper function for SAC agent's action selection"""
    return agent.select_action(state, evaluate=evaluate)

def sac_policy(state, agent, evaluate=True):
    """
    A policy function for SAC that takes the state and returns an action
    
    Parameters:
      - state: the current state (already obtained via env.get_state())
      - agent: the SAC agent
      - evaluate (bool): whether to use evaluation mode
      
    Returns:
      - action: the action selected by the SAC agent
    """
    return sac_select_action(state, agent, evaluate=evaluate)

def simulate_trial_sac(env_params, agent, num_sim=100, seed_start=0, track_metrics=True, parallel=False):
    """Simulate trial with SAC agent"""
    sac_policy_with_agent = partial(sac_policy, agent=agent)
    return simulate_trial_base(
        sac_policy_with_agent,
        env_params, num_sim, seed_start, 
        track_metrics, agent=agent, policy_name="sac", parallel=parallel
    )

# Function to run comparative analysis of multiple policies
def compare_policies(env_params, policies, num_sim=100, seed_start=0, track_metrics=True, parallel=False):
    """
    Run a comparative analysis of multiple policies on the same environment
    
    Parameters:
    -----------
    env_params : dict
        Parameters for the environment
    policies : dict
        Dictionary mapping policy names to (policy_fn, policy_args) tuples
    num_sim : int
        Number of simulations to run for each policy
    seed_start : int
        Starting seed for random number generation
    track_metrics : bool
        Whether to track additional metrics
    parallel : bool
        Whether to run simulations in parallel
        
    Returns:
    --------
    dict
        Dictionary mapping policy names to their results
    """
    results = {}
    
    for policy_name, (policy_fn, policy_args) in policies.items():
        print(f"\nEvaluating policy: {policy_name}")
        
        # Handle different policy types
        if policy_name == 'sac':
            # SAC agent requires special handling
            agent = policy_args
            results[policy_name] = simulate_trial_sac(
                env_params, agent, num_sim, seed_start + len(results), 
                track_metrics, parallel
            )
        elif policy_name == 'ideal':
            # Ideal policy has a special function
            results[policy_name] = simulate_trial_ideal(
                env_params, num_sim, seed_start + len(results), 
                track_metrics, parallel
            )
        else:
            # Heuristic policies
            results[policy_name] = simulate_trial_heuristic(
                env_params, policy_fn, num_sim, seed_start + len(results),
                track_metrics, parallel
            )
    
    # Perform statistical significance tests between policies
    if len(policies) > 1:
        results['statistical_tests'] = {}
        policy_names = list(policies.keys())
        
        for i in range(len(policy_names)):
            for j in range(i+1, len(policy_names)):
                p1 = policy_names[i]
                p2 = policy_names[j]
                
                # Get success proportions
                p1_success = results[p1]['success_proportions']
                p2_success = results[p2]['success_proportions']
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(p1_success, p2_success)
                
                # Calculate effect size (Cohen's d)
                effect_size = (np.mean(p1_success) - np.mean(p2_success)) / np.sqrt(
                    (np.std(p1_success)**2 + np.std(p2_success)**2) / 2
                )
                
                results['statistical_tests'][f'{p1}_vs_{p2}'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < 0.05
                }
    
    return results

def adapt_hyperparameters_to_environment(env_params):
    """
    Adapt SAC hyperparameters based on environment parameters
    
    Parameters:
    -----------
    env_params : dict
        Environment parameters
        
    Returns:
    --------
    hyperparams : dict
        Adapted hyperparameters
    """
    # Extract relevant environment parameters
    N = env_params['N']
    T = env_params['T']
    trial_size = N * T
    
    # Base hyperparameters
    hyperparams = {
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'hidden_dim': 128,
        'alpha': 0.2,
        'lambda_aux': 0,
        'auto_alpha': True,
        'target_entropy': -1.0
    }
    
    # Adjust learning rates based on trial size
    if trial_size > 500:
        # Smaller learning rates for larger trials
        hyperparams['actor_lr'] = 1e-4
        hyperparams['critic_lr'] = 1e-4
    elif trial_size < 100:
        # Larger learning rates for smaller trials
        hyperparams['actor_lr'] = 5e-4
        hyperparams['critic_lr'] = 5e-4
    
    # Adjust network size based on trial complexity
    if trial_size > 300:
        hyperparams['hidden_dim'] = 256
    elif trial_size < 100:
        hyperparams['hidden_dim'] = 64
    
    # Adjust discount factor based on trial length
    if T > 20:
        hyperparams['gamma'] = 0.995  # Higher gamma for longer horizons
    elif T < 5:
        hyperparams['gamma'] = 0.98   # Lower gamma for shorter horizons
    
    # Adjust auxiliary loss weight based on problem difficulty
    diff = abs(env_params['init_alpha_exp'] / (env_params['init_alpha_exp'] + env_params['init_beta_exp']) - 
               env_params['init_alpha_ctrl'] / (env_params['init_alpha_ctrl'] + env_params['init_beta_ctrl']))
    
    if diff < 0.1:
        # For similar arms, increase exploration
        hyperparams['lambda_aux'] = 0.0  # Reduce auxiliary loss influence
        hyperparams['target_entropy'] = -0.7  # Higher entropy
    elif diff > 0.3:
        # For very different arms, reduce exploration
        hyperparams['lambda_aux'] = 0.0  # Increase auxiliary loss influence
        hyperparams['target_entropy'] = -1.5  # Lower entropy
    
    return hyperparams
