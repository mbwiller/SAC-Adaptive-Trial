# Training Functions for SAC Clinical Trial Optimization
def train_sac_with_curriculum(env_params, agent, num_episodes, batch_size=64, 
                              eval_frequency=50, early_stopping_patience=None,
                              curriculum_schedule=None, validation_freq=None,
                              validation_episodes=10, smooth_transitions=True):
    """
    Train the SAC agent using curriculum learning with enhanced features
    
    Parameters:
    -----------
    env_params : dict
        Parameters for the environment
    agent : SACAgent
        The agent to train
    num_episodes : int
        Total number of episodes to train
    batch_size : int
        Batch size for updates
    eval_frequency : int
        How often to evaluate and print progress
    early_stopping_patience : int or None
        Number of evaluations with no improvement before stopping
    curriculum_schedule : list or None
        Custom curriculum schedule, if None use default
    validation_freq : int or None
        How often to perform validation on separate environment
    validation_episodes : int
        Number of episodes to run for validation
    smooth_transitions : bool
        Whether to smooth transitions between curriculum phases
    """
    # Default curriculum phases if not provided
    if curriculum_schedule is None:
        curriculum_schedule = [
            {'T_factor': 0.25, 'episodes': num_episodes // 4, 'reward_scale': 1.5},
            {'T_factor': 0.5, 'episodes': num_episodes // 4, 'reward_scale': 1.2},
            {'T_factor': 0.75, 'episodes': num_episodes // 4, 'reward_scale': 1.1},
            {'T_factor': 1.0, 'episodes': num_episodes // 4, 'reward_scale': 1.0}
        ]
    
    base_T = env_params['T']
    all_episode_rewards = []
    validation_history = []
    base_lambda_tvd = env_params.get('lambda_tvd', 0.1)
    base_lambda_explore = env_params.get('lambda_explore', 0.05)
    
    # Create a validation environment
    validation_env = None
    if validation_freq is not None:
        validation_env = EnhancedTrialEnv(**env_params)
    
    last_env = None
    
    for phase_idx, phase in enumerate(curriculum_schedule):    
        # Create environment for this phase
        curr_params = env_params.copy()
        curr_params['T'] = max(1, int(base_T * phase['T_factor']))
        
        # Scale information gain and exploration rewards for earlier phases
        reward_scale = phase.get('reward_scale', 1.0)
        curr_params['lambda_tvd'] = base_lambda_tvd * reward_scale
        curr_params['lambda_explore'] = base_lambda_explore * (1 + (1 - phase['T_factor']))
        
        env = EnhancedTrialEnv(**curr_params)
        
        # For smooth transitions: if we have a previous env, initialize state with previous knowledge
        if smooth_transitions and last_env is not None:
            env.alpha_exp = last_env.alpha_exp
            env.beta_exp = last_env.beta_exp
            env.alpha_ctrl = last_env.alpha_ctrl
            env.beta_ctrl = last_env.beta_ctrl
            # Scale allocation history to new trial length
            scale_factor = curr_params['T'] / last_env.T
            env.allocation_history_exp = int(last_env.allocation_history_exp * scale_factor)
            env.allocation_history_ctrl = int(last_env.allocation_history_ctrl * scale_factor)
        
        print(f"Training phase {phase_idx+1}/{len(curriculum_schedule)} with T={curr_params['T']} for {phase['episodes']} episodes")
        print(f"  Info gain scale: {curr_params['lambda_tvd']:.3f}, Exploration scale: {curr_params['lambda_explore']:.3f}")
        
        phase_rewards = train_sac(
            env, agent, phase['episodes'], 
            max_steps_per_episode=curr_params['T'], 
            batch_size=batch_size, 
            eval_frequency=eval_frequency,
            early_stopping_patience=early_stopping_patience,
            validation_env=validation_env if validation_freq else None,
            validation_freq=validation_freq,
            validation_episodes=validation_episodes
        )
        
        all_episode_rewards.extend(phase_rewards['training'])
        if 'validation' in phase_rewards:
            validation_history.extend(phase_rewards['validation'])
        
        # Keep reference to current env for next phase
        last_env = env
    
    return {
        'training_rewards': all_episode_rewards,
        'validation_rewards': validation_history if validation_freq else None,
    }

def train_sac(env, agent, num_episodes, max_steps_per_episode, batch_size=64, 
              eval_frequency=50, early_stopping_patience=None, early_stopping_threshold=0.01,
              validation_env=None, validation_freq=None, validation_episodes=10):
    """
    Enhanced training function with separate validation, improved early stopping
    and better tracking of performance metrics
    """
    start_time = time.time()
    episode_rewards = []
    evaluation_history = []
    validation_rewards = []
    
    best_eval_reward = -float('inf')
    patience_counter = 0
    
    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        step = 0
        episode_successes = 0
        
        while not done and step < max_steps_per_episode:
            # Select and execute action
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done = env.step(action, planning_mode=True)
            
            # Store transition
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            ep_reward += reward
            step += 1
            
            # Update agent
            agent.update(batch_size)
        
        episode_rewards.append(ep_reward)
        
        # Calculate success proportion for this episode
        success_proportion = env.total_successes / (env.N * env.T)
        
        # Print progress
        if (ep + 1) % eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-eval_frequency:])
            elapsed_time = time.time() - start_time
            
            # Track for history
            evaluation_history.append({
                'episode': ep + 1,
                'avg_reward': avg_reward,
                'time': elapsed_time,
                'success_proportion': success_proportion,
                'alpha': agent.alpha if hasattr(agent, 'auto_alpha') and agent.auto_alpha else None
            })
            
            print(f"Episode {ep+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Success Rate: {success_proportion:.4f}, Time: {elapsed_time:.2f}s")
            
            # Print current alpha if using automatic tuning
            if hasattr(agent, 'auto_alpha') and agent.auto_alpha:
                print(f"  Current alpha: {agent.alpha:.4f}")
            
            # Run validation if requested
            if validation_env is not None and validation_freq is not None and (ep + 1) % validation_freq == 0:
                eval_results = evaluate_agent(validation_env, agent, num_episodes=validation_episodes)
                val_success = eval_results['success_proportion'][0]  # Extract mean
                val_std = eval_results['success_proportion'][1]      # Extract std error
                validation_rewards.append({
                    'episode': ep + 1,
                    'success_proportion': val_success,
                    'std_error': val_std
                })
                print(f"  Validation Success Rate: {val_success:.4f} ± {val_std:.4f}")
                
                # Use validation performance for early stopping if available
                eval_reward = val_success
            else:
                # Otherwise use episode rewards as proxy
                eval_reward = avg_reward
            
            # Check for early stopping with improved threshold
            if early_stopping_patience is not None:
                # Only consider it an improvement if it exceeds by the threshold
                if eval_reward > best_eval_reward * (1 + early_stopping_threshold):
                    best_eval_reward = eval_reward
                    patience_counter = 0
                    print(f"  New best performance: {eval_reward:.4f}")
                else:
                    patience_counter += 1
                    print(f"  No improvement for {patience_counter} evaluations")
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at episode {ep+1}")
                    break
    
    results = {
        'training': episode_rewards,
        'evaluations': evaluation_history,
    }
    
    if validation_rewards:
        results['validation'] = validation_rewards
    
    return results
