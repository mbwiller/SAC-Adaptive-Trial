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
