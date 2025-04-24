def tune_hyperparameters(env_params, trials=20, episodes_per_trial=1000,
                         tuning_method="bayesian", eval_metrics=None,
                         num_evaluation_episodes=20, search_seed=42,
                         save_results=True, parallel=True, max_workers=None,
                         create_plots=True, verbosity=2):

    """
    Enhanced hyperparameter tuning with Bayesian strategy
    
    Parameters:
    -----------
    env_params : dict
        Parameters for the environment
    trials : int
        Number of hyperparameter combinations to try
    episodes_per_trial : int
        Number of episodes to train each agent
    tuning_method : str
        Tuning method: "bayesian"
    eval_metrics : list or None
        Additional metrics to consider besides success proportion
    num_evaluation_episodes : int
        Number of episodes to use for evaluation
    search_seed : int
        Seed for reproducibility in parameter search
    save_results : bool
        Whether to save tuning results to disk
    parallel : bool
        Whether to use parallel processing for evaluation
    max_workers : int or None
        Maximum number of parallel workers, None for auto
    create_plots : bool
        Whether to create plots when saving results
    verbosity : int
        Verbosity level (0=minimal, 1=normal, 2=detailed)
        
    Returns:
    --------
    dict
        Best hyperparameters found
    """
    # Set up random seed for reproducibility
    np.random.seed(search_seed)
    rng = np.random.RandomState(search_seed)
    
    # Set default evaluation metrics if not provided
    if eval_metrics is None:
        eval_metrics = ['success_proportion', 'exp_allocation', 'prob_exp_better']
    
    # Create environment
    env = EnhancedTrialEnv(**env_params)
    state_dim = len(env.get_state())
    
    # Define hyperparameter space
    param_space = {
        'actor_lr': [7.5e-5, 1e-4, 2e-4],
        'critic_lr': [7.7e-5, 1e-4, 2e-4],
        'gamma': [0.99, 0.995, 0.999],
        'tau': [0.003, 0.005, 0.007],
        'hidden_dim': [256],
        'alpha': [0.2, 0.3],
        'lambda_aux': [0, 0.05, 0.075],
        'auto_alpha': [True],  # Include auto temperature tuning
        'target_entropy': [-0.5, -0.7]  # Different target entropy values
    }
    
    # Define known interactions
    param_interactions = [
        # Learning rates should be related 
        ('actor_lr', 'critic_lr'),
        # Network size affects learning rate
        ('hidden_dim', 'actor_lr'),
        ('hidden_dim', 'critic_lr'),
        # Auto alpha interacts with target entropy
        ('auto_alpha', 'target_entropy'),
        # Auxiliary loss weight interacts with alpha
        ('lambda_aux', 'alpha')
    ]
    
    print(f"Starting hyperparameter tuning with {tuning_method} method...")
    print(f"Will try {trials} parameter combinations, training each for {episodes_per_trial} episodes")
    
    # Setup result storage
    results_history = []
    trial_configs = []
    model_performance = []

    # Determine how many random trials to run
    random_trials = min(trials, max(5, int(trials * 0.5)))  # Use at least 5 trials, up to 50% of total
    param_combinations = generate_random_configs(param_space, random_trials, rng)
    
    # Track best performers
    best_score = -float('inf')
    best_params = None
    best_metrics = None
    
    # Run random trials
    for trial, params in enumerate(param_combinations):
        # Ensure valid parameter combinations
        params = fix_param_interactions(params, param_interactions)
        
        try:
            print(f"\nTrial {trial+1}/{trials} - Testing parameters: {params}")
            start_time = time.time()
            
            # Create agent
            agent = SACAgent(
                state_dim=state_dim,
                actor_lr=params['actor_lr'],
                critic_lr=params['critic_lr'],
                gamma=params['gamma'],
                tau=params['tau'],
                hidden_dim=params['hidden_dim'],
                alpha=params['alpha'] if not params.get('auto_alpha', False) else 0.2,
                lambda_aux=params['lambda_aux'],
                auto_alpha=params.get('auto_alpha', False),
                target_entropy=params.get('target_entropy', -1.0) if params.get('auto_alpha', False) else None,
                device=device
            )
            
            # Train agent using curriculum learning
            training_result = train_sac_with_curriculum(
                env_params=env_params,
                agent=agent,
                num_episodes=episodes_per_trial,
                batch_size=64,
                validation_freq=episodes_per_trial // 5  # Validate 5 times during training
            )
            
            # Comprehensive evaluation
            eval_result = simulate_trial_sac(
                env_params=env_params, 
                agent=agent, 
                num_sim=num_evaluation_episodes,
                track_metrics=True,
                parallel=parallel
            )
            
            # Extract primary metric (success proportion)
            success_prop = eval_result['success_proportion'][0]
            success_std = eval_result['success_proportion'][1]
            
            # Calculate multi-objective score if using multiple metrics
            if len(eval_metrics) > 1:
                # Get metrics values
                metrics_values = {}
                metric_weights = {
                    'success_proportion': 1.0,  # Primary metric
                    'info_gain': 0.3,           # Information gain
                    'exp_allocation': 0.2,      # Balanced allocations
                    'allocation_variability': -0.2,  # Less variability is better
                    'prob_exp_better': 0.3      # Confidence in results
                }
                
                # Collect available metrics
                for metric in eval_metrics:
                    if metric in eval_result:
                        if isinstance(eval_result[metric], tuple):
                            # If it's a tuple with (mean, std), use the mean
                            value = eval_result[metric][0]
                        else:
                            # Otherwise use the direct value
                            value = eval_result[metric]
                        metrics_values[metric] = value
                
                # Normalize and calculate combined score
                combined_score = success_prop  # Start with success proportion
                for metric, value in metrics_values.items():
                    if metric != 'success_proportion' and metric in metric_weights:
                        # Add weighted contributions of other metrics
                        combined_score += value * metric_weights.get(metric, 0.1)
            else:
                # Just use success proportion
                combined_score = success_prop
            
            # Store evaluation results
            metrics_summary = {
                'trial': trial+1,
                'success_proportion': success_prop,
                'success_std': success_std,
                'combined_score': combined_score,
                'training_time': time.time() - start_time
            }
            
            # Add other metrics to summary
            for metric in eval_metrics:
                if metric in eval_result and metric != 'success_proportion':
                    if isinstance(eval_result[metric], tuple):
                        metrics_summary[metric] = eval_result[metric][0]
                    else:
                        metrics_summary[metric] = eval_result[metric]
            
            # Print results
            print(f"Trial {trial+1} - Success proportion: {success_prop:.4f} ± {success_std:.4f}")
            print(f"          Combined score: {combined_score:.4f}")
            
            # Store results
            trial_configs.append(params)
            model_performance.append(metrics_summary)
            
            # Update best parameters
            if combined_score > best_score:
                best_score = combined_score
                best_params = params.copy()
                best_metrics = metrics_summary.copy()
                print(f"          New best parameters found!")
            
            # Store detailed history
            results_history.append({
                'params': params,
                'metrics': metrics_summary,
                'best_so_far': best_score == combined_score
            })
            
        except Exception as e:
            print(f"Error during trial {trial+1}: {e}")
            # Continue with next trial
    
    # Analyze parameter importance if we have enough data
    if len(model_performance) >= 5:
        print("\nCompleting hyperparameter search with best parameters from random trials")
        
        try:
            if len(model_performance) >= 10:
                print("\nAnalyzing parameter importance...")
                param_importance = analyze_parameter_importance(trial_configs, model_performance)
                
                # Print importance scores
                print("Parameter importance (higher scores = more important):")
                for param, importance in sorted(param_importance.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.4f}")
        except Exception as e:
            print(f"Error analyzing parameter importance: {e}")
    
    # Save results if requested
    if save_results:
        try:
            save_tuning_results(trial_configs, model_performance, results_history, 
                                param_space, env_params, best_params,
                                create_plots=create_plots, verbosity=verbosity)
        except Exception as e:
            print(f"Error saving tuning results: {e}")
    
    print(f"\nHyperparameter tuning complete.")
    print(f"Best parameters: {best_params}")
    if best_metrics:
        print(f"Best performance: {best_metrics['success_proportion']:.4f} ± {best_metrics['success_std']:.4f}")
    
    # If we didn't find any good parameters, fall back to adaptive ones
    if best_params is None:
        print("Warning: No valid parameters found. Using adaptive hyperparameters.")
        best_params = adapt_hyperparameters_to_environment(env_params)
        best_params['lambda_aux'] = 0
    
    return best_params

def fix_param_interactions(params, interactions):
    """
    Fix parameter interactions to ensure sensible combinations
    
    This handles known interactions between parameters to avoid invalid
    or suboptimal combinations.
    """
    params = params.copy()
    
    # Handle auto_alpha and target_entropy
    if 'auto_alpha' in params and not params['auto_alpha']:
        # If not using auto_alpha, target_entropy is irrelevant
        if 'target_entropy' in params:
            del params['target_entropy']
    
    # Ensure actor and critic learning rates are related
    if ('actor_lr', 'critic_lr') in interactions:
        if abs(np.log10(params['actor_lr']) - np.log10(params['critic_lr'])) > 1:
            # Ensure learning rates are within an order of magnitude
            params['critic_lr'] = params['actor_lr']
    
    # Adjust learning rates based on network size
    if ('hidden_dim', 'actor_lr') in interactions:
        if params['hidden_dim'] >= 256 and params['actor_lr'] > 5e-4:
            # For larger networks, use smaller learning rates
            params['actor_lr'] = 5e-4
        
    if ('hidden_dim', 'critic_lr') in interactions:
        if params['hidden_dim'] >= 256 and params['critic_lr'] > 5e-4:
            # For larger networks, use smaller learning rates
            params['critic_lr'] = 5e-4
    
    return params

def generate_random_configs(param_space, n_configs, rng=None):
    """Generate random hyperparameter configurations"""
    if rng is None:
        rng = np.random.RandomState()
        
    configs = []
    for _ in range(n_configs):
        config = {}
        for param, values in param_space.items():
            config[param] = rng.choice(values)
        configs.append(config)
    
    return configs

def analyze_parameter_importance(configs, performances):
    """
    Analyze which hyperparameters have the most impact on performance
    
    Returns a dictionary mapping parameter names to importance scores
    """
    # Prepare data
    X, y = prepare_data_for_modeling(configs, performances)
    
    # Extract parameter names
    param_names = set()
    for config in configs:
        param_names.update(config.keys())
    param_names = sorted(list(param_names))
    
    # Train a Random Forest to assess feature importance
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Map back to parameter names
    result = {}
    for i, param in enumerate(param_names):
        result[param] = importances[i]
    
    return result

def save_tuning_results(configs, performances, history, param_space, env_params, best_params, 
                       create_plots=True, verbosity=2):
    """
    Save hyperparameter tuning results to disk
    
    Parameters:
    -----------
    configs : list
        List of parameter configurations
    performances : list
        List of performance metrics
    history : list
        Detailed history of the tuning process
    param_space : dict
        The hyperparameter search space
    env_params : dict
        Environment parameters
    best_params : dict
        Best hyperparameters found
    create_plots : bool, default=True
        Whether to generate and save plots
    verbosity : int, default=1
        Level of output detail (0=minimal, 1=normal, 2=detailed)
    """
    os.makedirs('tuning_results_v22', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"tuning_results/hyperparameter_tuning_{timestamp}"
    
    try:
        serializable_configs = []
        for config in configs:
            serializable_config = {}
            for k, v in config.items():
                if isinstance(v, (int, float, str, bool, list, dict, tuple)):
                    serializable_config[k] = v
                else:
                    serializable_config[k] = str(v)
            serializable_configs.append(serializable_config)
        
        serializable_performances = []
        for perf in performances:
            serializable_perf = {}
            for k, v in perf.items():
                if isinstance(v, (int, float, str, bool, list, dict, tuple)):
                    serializable_perf[k] = v
                elif isinstance(v, np.ndarray):
                    serializable_perf[k] = v.tolist() if v.size > 1 else float(v)
                else:
                    serializable_perf[k] = str(v)
            serializable_performances.append(serializable_perf)
            
        serializable_history = []
        for entry in history:
            serializable_entry = {}
            for k, v in entry.items():
                if k == 'params':
                    serializable_entry[k] = {}
                    for pk, pv in v.items():
                        if isinstance(pv, (int, float, str, bool, list, dict, tuple)):
                            serializable_entry[k][pk] = pv
                        else:
                            serializable_entry[k][pk] = str(pv)
                elif k == 'metrics':
                    serializable_entry[k] = {}
                    for mk, mv in v.items():
                        if isinstance(mv, (int, float, str, bool, list, dict, tuple)):
                            serializable_entry[k][mk] = mv
                        elif isinstance(mv, np.ndarray):
                            serializable_entry[k][mk] = mv.tolist() if mv.size > 1 else float(mv)
                        else:
                            serializable_entry[k][mk] = str(mv)
                else:
                    if isinstance(v, (int, float, str, bool, list, dict, tuple)):
                        serializable_entry[k] = v
                    elif isinstance(v, np.ndarray):
                        serializable_entry[k] = v.tolist() if v.size > 1 else float(v)
                    else:
                        serializable_entry[k] = str(v)
            serializable_history.append(serializable_entry)
        
        serializable_best_params = {}
        if best_params:
            for k, v in best_params.items():
                if isinstance(v, (int, float, str, bool, list, dict, tuple)):
                    serializable_best_params[k] = v
                else:
                    serializable_best_params[k] = str(v)
        
        results = {
            'timestamp': timestamp,
            'param_space': param_space,
            'env_params': env_params,
            'best_params': serializable_best_params,
            'configs': serializable_configs,
            'performances': serializable_performances,
            'history': serializable_history
        }
    
        with open(f"{filename}.json", 'w') as f:
            json.dump(results, f)
    
    except Exception as e:
        if verbosity >= 1:
            print(f"Error saving JSON results: {e}")
        try:
            import simplejson
            with open(f"{filename}.json", 'w') as f:
                simplejson.dump(results, f, ignore_nan=True)
            if verbosity >= 1:
                print("Results saved using simplejson")
        except:
            try:
                import pickle
                with open(f"{filename}.pkl", 'wb') as f:
                    pickle.dump(results, f)
                if verbosity >= 1:
                    print(f"Results saved as pickle to {filename}.pkl")
            except Exception as e2:
                if verbosity >= 1:
                    print(f"Could not save results: {e2}")
    
    if verbosity >= 1:
        print(f"Results saved to {filename}.json")
    
    if create_plots:
        try:
            if verbosity >= 1:
                print("Generating performance plots...")
            
            # Create trial vs. performance plot
            plt.figure(figsize=(10, 6))
            
            trials = [p['trial'] for p in performances]
            success_props = [p['success_proportion'] for p in performances]
            
            plt.plot(trials, success_props, 'o-', label='Success Proportion')
            
            if 'combined_score' in performances[0]:
                combined_scores = [p['combined_score'] for p in performances]
                plt.plot(trials, combined_scores, 's-', label='Combined Score')
            
            plt.xlabel('Trial')
            plt.ylabel('Performance')
            plt.title('Hyperparameter Tuning Performance')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"{filename}_performance.png")
            plt.close()
            
            if verbosity >= 2:
                print(f"  Generated overall performance plot: {filename}_performance.png")
            
            # Create parameter correlation plots for key parameters
            key_params = ['actor_lr', 'hidden_dim', 'gamma', 'alpha']
            key_params = [p for p in key_params if p in configs[0]]
            
            for param in key_params:
                plt.figure(figsize=(8, 5))
                
                param_values = [config[param] for config in configs]
                
                plt.scatter(param_values, success_props)
                plt.xlabel(param)
                plt.ylabel('Success Proportion')
                plt.title(f'Effect of {param} on Performance')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                if len(param_values) > 2 and all(isinstance(x, (int, float)) for x in param_values):
                    try:
                        z = np.polyfit(param_values, success_props, 1)
                        p = np.poly1d(z)
                        plt.plot(sorted(param_values), p(sorted(param_values)), "r--")
                    except:
                        pass
                
                plt.savefig(f"{filename}_{param}.png")
                plt.close()
                
                if verbosity >= 2:
                    print(f"  Generated parameter plot for {param}: {filename}_{param}.png")
            
            if verbosity >= 1:
                print("Plot generation complete.")
                
        except Exception as e:
            if verbosity >= 1:
                print(f"Error creating plots: {e}")
    
    return filename
