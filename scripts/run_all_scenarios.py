# Main execution for SAC clinical trial model
if __name__ == "__main__":
 
    SEED = 77
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Resource management to prevent "Too many open files" errors
    try:
        gc.collect()
        
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Current file descriptor limits: soft={soft}, hard={hard}")
        
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 4096), hard))
            print(f"File descriptor limit set to {min(hard, 4096)}")
        except (ValueError, OSError) as e:
            print(f"Warning: Could not adjust file descriptor limits: {e}")
            print("Will use more conservative parallel settings.")
    except Exception as e:
        print(f"Warning: Resource management setup failed: {e}")
    
    os.makedirs("results_v22F", exist_ok=True)
    os.makedirs("results/plots_v22F", exist_ok=True)
    os.makedirs("results/models_v22F", exist_ok=True)
    os.makedirs("tuning_results_v22F", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_info(message):
        print(message)
        with open(f"results/log_{timestamp}.txt", "a") as f:
            f.write(message + "\n")
    
    log_info(f"Starting SAC clinical trial model evaluation at {timestamp}")
    log_info("=" * 80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_info(f"Using device: {device}")
    
    test_scenarios = [
        {
            'name': 'Subtle Edge, Uninformative Priors',
            'params': {
                'N': 40,
                'T': 20,
                'init_alpha_exp': 1,
                'init_beta_exp': 1,
                'init_alpha_ctrl': 1,
                'init_beta_ctrl': 1,
                'p_true_exp': 0.53,   # 3% difference
                'p_true_ctrl': 0.50,
                'lambda_tvd': 0.12,
                'lambda_explore': 0.05
            }
        },
        {
            'name': 'Dynamic Treatment Effect',
            'params': {
                'N': 25,              
                'T': 20,              
                'init_alpha_exp': 3,  # Weak priors
                'init_beta_exp': 7,
                'init_alpha_ctrl': 5,
                'init_beta_ctrl': 5,
                'p_true_exp': 0.65,   # Strong advantage (0.15) despite poor prior
                'p_true_ctrl': 0.50,
                'lambda_tvd': 0.15,   # Higher info gain reward
                'lambda_explore': 0.1
            }
        },
        {
            'name': 'High Uncertainty - Large Cohort',
            'params': {
                'N': 100,             # Large cohort
                'T': 8,               # Relatively short trial
                'init_alpha_exp': 1,  # Very uninformative priors
                'init_beta_exp': 1,
                'init_alpha_ctrl': 1,
                'init_beta_ctrl': 1,
                'p_true_exp': 0.70,   # Clear advantage (0.20)
                'p_true_ctrl': 0.50,
                'lambda_tvd': 0.1,
                'lambda_explore': 0.05
            }
        },
        {
            'name': 'Barely Superior Treatment',
            'params': {
                'N': 50,              
                'T': 12,              
                'init_alpha_exp': 10, # Strong confidence in experimental
                'init_beta_exp': 10,
                'init_alpha_ctrl': 10,
                'init_beta_ctrl': 10,
                'p_true_exp': 0.52,   # Very small advantage (0.02)
                'p_true_ctrl': 0.50,
                'lambda_tvd': 0.08,
                'lambda_explore': 0.08
            }
        },
        {
            'name': 'Strongly Misleading Prior',
            'params': {
                'N': 40,              
                'T': 18,              
                'init_alpha_exp': 2,  # Prior suggests exp is worse
                'init_beta_exp': 8,
                'init_alpha_ctrl': 8,
                'init_beta_ctrl': 2,
                'p_true_exp': 0.65,   # But exp is actually better
                'p_true_ctrl': 0.40,
                'lambda_tvd': 0.12,
                'lambda_explore': 0.07
            }
        },
        {
            'name': 'Clear Advantage, Longer Trial',
            'params': {
                'N': 30,
                'T': 25,
                'init_alpha_exp': 4,   # Moderately informative priors
                'init_beta_exp': 6,
                'init_alpha_ctrl': 5,
                'init_beta_ctrl': 5,
                'p_true_exp': 0.60,    # 5% difference
                'p_true_ctrl': 0.55,
                'lambda_tvd': 0.1,
                'lambda_explore': 0.05
            }
        },
        {
            'name': 'Large Cohort, Low-Effect',
            'params': {
                'N': 100,
                'T': 8,
                'init_alpha_exp': 1,
                'init_beta_exp': 1,
                'init_alpha_ctrl': 1,
                'init_beta_ctrl': 1,
                'p_true_exp': 0.57,   # 2% absolute improvement
                'p_true_ctrl': 0.55,
                'lambda_tvd': 0.1,
                'lambda_explore': 0.05
            }
        },
        {
            'name': 'Modest Improvement in High-Risk Population',
            'params': {
                'N': 40,               # Moderate cohort size
                'T': 15,               # Sufficient trial duration
                'init_alpha_exp': 3,   # Slightly informative priors
                'init_beta_exp': 7,
                'init_alpha_ctrl': 4,
                'init_beta_ctrl': 6,
                'p_true_exp': 0.58,    # Around 8 percentage points better
                'p_true_ctrl': 0.50,
                'lambda_tvd': 0.1,
                'lambda_explore': 0.05
            }
        },
        {
            'name': 'Minimal Clinically Important Difference',
            'params': {
                'N': 60,               # Slightly larger to improve sensitivity
                'T': 25,               # Longer duration for incremental updates
                'init_alpha_exp': 2,   # Uninformative priors to let data drive learning
                'init_beta_exp': 2,
                'init_alpha_ctrl': 2,
                'init_beta_ctrl': 2,
                'p_true_exp': 0.54,    # 4% absolute improvement
                'p_true_ctrl': 0.50,
                'lambda_tvd': 0.12,
                'lambda_explore': 0.04
            }
        },
        {
            'name': 'Low Event Rate Scenario',
            'params': {
                'N': 80,               # Larger cohort to compensate for low event rate
                'T': 20,               # Longer trial to observe enough events
                'init_alpha_exp': 2,   # Uninformative priors
                'init_beta_exp': 8,
                'init_alpha_ctrl': 2,
                'init_beta_ctrl': 8,
                'p_true_exp': 0.20,    # Low overall success rates with a 3-4% gap
                'p_true_ctrl': 0.17,
                'lambda_tvd': 0.11,
                'lambda_explore': 0.05
            }
        }
    ]
    
    # Configuration with tunable options
    config = {
        'training_episodes': 2000,   # Total training episodes
        'batch_size': 64,
        'eval_episodes': 500,          # Episodes for policy evaluation
        'simulation_count': 500,       # Number of simulation runs per policy
        'tune_hyperparams': True,     # Whether to tune hyperparameters
        'tuning_trials': 10,          # Number of hyperparameter combinations to try
        'auto_alpha': True,           # Auto temperature adjustment in SAC
        'hidden_dim': 128,            # Default hidden dimension for neural networks
        'use_fixed_outcomes': False,  # Whether to use fixed outcomes for fair comparison
        'parallel': False,            # Whether to use parallel processing
        'max_workers': 2,             # Maximum number of parallel workers
        'save_models': False,          # Whether to save trained models
        'advanced_visualization': True, # Whether to use advanced visualization functions
        'verbosity': 2                # Output verbosity level (0=minimal, 1=normal, 2=detailed)
    }
    
    all_results = []
    
    for scenario_idx, scenario in enumerate(test_scenarios):
        scenario_name = scenario['name']
        env_params = scenario['params']
        
        log_info(f"\n\n{'-' * 40}")
        log_info(f"Scenario {scenario_idx+1}/{len(test_scenarios)}: {scenario_name}")
        log_info(f"Parameters: {env_params}")
        
        env = EnhancedTrialEnv(**env_params)
        state_dim = len(env.get_state())
        
        log_info(f"Environment created with state dimension: {state_dim}")
        
        # Step 1: Hyperparameter Tuning
        if config['tune_hyperparams']:
            log_info(f"Running hyperparameter tuning with {config['tuning_trials']} trials...")
            
            try:
                best_hyperparams = tune_hyperparameters(
                    env_params=env_params,
                    trials=config['tuning_trials'],
                    episodes_per_trial=500,
                    tuning_method="bayesian",
                    eval_metrics=['success_proportion', 'prob_exp_better', 'exp_allocation'],
                    num_evaluation_episodes=10,
                    search_seed=SEED + scenario_idx,
                    save_results=True,
                    parallel=config['parallel'],
                    max_workers=config['max_workers'],
                    create_plots=config['advanced_visualization'],
                    verbosity=config['verbosity']
                )
            except Exception as e:
                log_info(f"Error during hyperparameter tuning: {str(e)}")
                log_info("Falling back to adaptive hyperparameters")
                best_hyperparams = adapt_hyperparameters_to_environment(env_params)
                best_hyperparams['lambda_aux'] = 0
            
            log_info(f"Best hyperparameters found: {best_hyperparams}")
        else:
            # Use adaptive hyperparameters instead of tuning
            best_hyperparams = adapt_hyperparameters_to_environment(env_params)
            best_hyperparams['lambda_aux'] = 0
            log_info(f"Using adaptive hyperparameters: {best_hyperparams}")
        
        # Step 2: Create the SAC agent with optimized hyperparameters
        log_info("Creating SAC agent with optimized hyperparameters...")
        agent = SACAgent(
            state_dim=state_dim,
            actor_lr=best_hyperparams['actor_lr'],
            critic_lr=best_hyperparams['critic_lr'],
            gamma=best_hyperparams['gamma'],
            tau=best_hyperparams['tau'],
            hidden_dim=best_hyperparams.get('hidden_dim', config['hidden_dim']),
            alpha=best_hyperparams['alpha'],
            lambda_aux=best_hyperparams.get('lambda_aux', 0),
            auto_alpha=best_hyperparams.get('auto_alpha', config['auto_alpha']),
            target_entropy=best_hyperparams.get('target_entropy', -1.0),
            device=device
        )

        # Step 3: Train the agent with curriculum learning
        curriculum_schedule = [
            {'T_factor': 0.25, 'episodes': config['training_episodes'] // 4, 'reward_scale': 1.5},
            {'T_factor': 0.5, 'episodes': config['training_episodes'] // 4, 'reward_scale': 1.2},
            {'T_factor': 0.75, 'episodes': config['training_episodes'] // 4, 'reward_scale': 1.1},
            {'T_factor': 1.0, 'episodes': config['training_episodes'] // 4, 'reward_scale': 1.0}
        ]
        
        log_info(f"Training agent with curriculum learning ({config['training_episodes']} episodes)...")
        train_start = time.time()
        
        training_result = train_sac_with_curriculum(
            env_params=env_params,
            agent=agent,
            num_episodes=config['training_episodes'],
            batch_size=config['batch_size'],
            eval_frequency=10,
            curriculum_schedule=curriculum_schedule,
            validation_freq=config['training_episodes'] // 5,
            validation_episodes=config['eval_episodes']
        )
        
        train_time = time.time() - train_start
        log_info(f"Training completed in {train_time:.2f} seconds")

        try:
            oracle_allocation, oracle_success, oracle_outcomes_exp, oracle_outcomes_ctrl, oracle_metadata = run_oracle_benchmark(
                env_params['N'], env_params['T'], env_params['p_true_exp'], env_params['p_true_ctrl'], seed=SEED)
            log_info(f"Oracle benchmark: Best allocation = {oracle_allocation}, Success proportion = {oracle_success:.4f}")
        except Exception as e:
            log_info(f"Error running oracle benchmark: {e}")
        
        if config['save_models']:
            model_path = f"results/models/sac_agent_{scenario_idx+1}_{timestamp}.pt"
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic1_state_dict': agent.critic1.state_dict(),
                'critic2_state_dict': agent.critic2.state_dict(),
                'hyperparams': best_hyperparams,
                'env_params': env_params,
                'scenario_name': scenario_name
            }, model_path)
            log_info(f"Model saved to {model_path}")
        
        # Step 4: Advanced Evaluation
        log_info("\nRunning policy evaluation...")     
        
        policies_to_compare = {
            'fixed': (heuristic_fixed, None),
            'greedy': (heuristic_greedy, None),
            'pl': (heuristic_PL, None),
            'ideal': (ideal_policy, None),
            'sac': (sac_planning_policy, agent)
        }
        
        try:
            comparison_results = compare_policies(
                env_params=env_params,
                policies=policies_to_compare,
                num_sim=config['simulation_count'],
                seed_start=SEED + scenario_idx,
                track_metrics=True,
                parallel=config['parallel']
            )
            
            sac_success = comparison_results['sac']['success_proportion'][0]
            fixed_success = comparison_results['fixed']['success_proportion'][0]
            greedy_success = comparison_results['greedy']['success_proportion'][0]
            pl_success = comparison_results['pl']['success_proportion'][0]
            ideal_success = comparison_results['ideal']['success_proportion'][0]
            
            sac_alloc = comparison_results['sac']['exp_allocation'][0]
            fixed_alloc = comparison_results['fixed']['exp_allocation'][0]
            greedy_alloc = comparison_results['greedy']['exp_allocation'][0]
            pl_alloc = comparison_results['pl']['exp_allocation'][0]
            ideal_alloc = comparison_results['ideal']['exp_allocation'][0]
            
            for policy_name in ['sac', 'fixed', 'greedy', 'pl', 'ideal']:
                log_info(f"{policy_name.upper()} Policy Results:")
                log_info(f"  Success Rate: {comparison_results[policy_name]['success_proportion'][0]:.4f} ± {comparison_results[policy_name]['success_proportion'][1]:.4f}")
                log_info(f"  Exp Allocation: {comparison_results[policy_name]['exp_allocation'][0]:.4f}")
                info_gain = comparison_results[policy_name].get('info_gain', (0.0, 0.0))[0]
                log_info(f"  Information Gain: {info_gain:.4f}")
            
            # Calculate performance improvements
            imp_fixed = ((sac_success - fixed_success) / fixed_success * 100) if fixed_success > 0 else 0
            imp_greedy = ((sac_success - greedy_success) / greedy_success * 100) if greedy_success > 0 else 0
            imp_pl = ((sac_success - pl_success) / pl_success * 100) if pl_success > 0 else 0
            gap_to_ideal = ((ideal_success - sac_success) / ideal_success * 100) if ideal_success > 0 else 0
            
            # Log performance improvements
            log_info("\nPerformance Improvements:")
            log_info(f"  vs Fixed: {imp_fixed:.2f}%")
            log_info(f"  vs Greedy: {imp_greedy:.2f}%")
            log_info(f"  vs PL: {imp_pl:.2f}%")
            log_info(f"  Gap to Ideal: {gap_to_ideal:.2f}%")
            
            # Step 5: Create SAC trajectory visualization
            if 'sac' in comparison_results and 'trajectories' in comparison_results['sac']:
                try:
                    log_info("Generating SAC trajectory visualization...")
                    if comparison_results['sac']['trajectories'] and len(comparison_results['sac']['trajectories']) > 0:
                        trajectory = comparison_results['sac']['trajectories'][0]
                        plot_file = plot_sac_trajectory(
                            trajectories=trajectory,
                            env_params=env_params,
                            save_prefix=f"sac_trajectory_{scenario_idx+1}_{timestamp}",
                            scenario_name=scenario_name,
                            scenario_idx=scenario_idx,
                            timestamp=timestamp
                        )
                        log_info(f"SAC trajectory plot saved to {plot_file}")
                    else:
                        log_info("No trajectory data available for visualization")
                except Exception as e:
                    log_info(f"Error creating trajectory plot: {str(e)}")
            
            # Step 6: Create DataFrame for this scenario and visualize results
            try:
                log_info("Creating advanced visualizations...")
                scenario_df = pd.DataFrame({
                    'scenario': [scenario_name],
                    'N': [env_params['N']],
                    'T': [env_params['T']],
                    'p_diff': [env_params['p_true_exp'] - env_params['p_true_ctrl']],
                    'prior_diff': [(env_params['init_alpha_exp']/(env_params['init_alpha_exp']+env_params['init_beta_exp'])) -
                                 (env_params['init_alpha_ctrl']/(env_params['init_alpha_ctrl']+env_params['init_beta_ctrl']))],
                    'sac_success': [sac_success],
                    'fixed_success': [fixed_success],
                    'greedy_success': [greedy_success],
                    'pl_success': [pl_success],
                    'ideal_success': [ideal_success],
                    'sac_alloc': [sac_alloc],
                    'fixed_alloc': [fixed_alloc],
                    'greedy_alloc': [greedy_alloc],
                    'pl_alloc': [pl_alloc],
                    'ideal_alloc': [ideal_alloc],
                    'sac_imp_fixed': [imp_fixed],
                    'sac_imp_greedy': [imp_greedy],
                    'sac_imp_pl': [imp_pl],
                    'gap_to_ideal': [gap_to_ideal]
                })
                
                if config['advanced_visualization']:
                    visualize_results(
                        results_df=scenario_df,
                        save_prefix=f"{scenario_idx+1}_{timestamp}"
                    )
                    log_info("Advanced visualizations created")
            except Exception as e:
                log_info(f"Error in scenario visualization: {str(e)}")
            
            # Step 7: Store comprehensive results for this scenario
            scenario_results = {
                'scenario': scenario_name,
                'N': env_params['N'],
                'T': env_params['T'],
                'p_diff': env_params['p_true_exp'] - env_params['p_true_ctrl'],
                'prior_diff': (env_params['init_alpha_exp']/(env_params['init_alpha_exp']+env_params['init_beta_exp'])) -
                             (env_params['init_alpha_ctrl']/(env_params['init_alpha_ctrl']+env_params['init_beta_ctrl'])),
                'train_time': train_time,
                'sac_success': sac_success,
                'fixed_success': fixed_success,
                'greedy_success': greedy_success,
                'pl_success': pl_success,
                'ideal_success': ideal_success,
                'sac_alloc': sac_alloc,
                'fixed_alloc': fixed_alloc,
                'greedy_alloc': greedy_alloc,
                'pl_alloc': pl_alloc,
                'ideal_alloc': ideal_alloc,
                'sac_imp_fixed': imp_fixed,
                'sac_imp_greedy': imp_greedy,
                'sac_imp_pl': imp_pl,
                'gap_to_ideal': gap_to_ideal
            }
            
            all_results.append(scenario_results)
            
        except Exception as e:
            log_info(f"Error in policy evaluation: {str(e)}")
            log_info("Continuing to next scenario...")
            continue
        
    # Step 8: Generate Final Summary and Save Results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        results_df.to_csv(f"results/comprehensive_results_{timestamp}.csv", index=False)
        log_info(f"\nResults saved to results/comprehensive_results_{timestamp}.csv")
        
        if len(all_results) > 1 and config['advanced_visualization']:
            try:
                log_info("Generating cross-scenario comparisons...")
                visualize_results(results_df, save_prefix=f"all_scenarios_{timestamp}")
                log_info("Cross-scenario visualizations created")
            except Exception as e:
                log_info(f"Error in cross-scenario visualization: {str(e)}")
    
    log_info("\nEvaluation complete! Check the results directory for output files and plots.")
