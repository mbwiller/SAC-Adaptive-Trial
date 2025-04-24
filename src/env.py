# TVD computation
@lru_cache(maxsize=1024)
def compute_tvd(alpha1, beta1, alpha2, beta2, grid_points=100):
    """Compute Total Variation Distance between two beta distributions"""
    x = np.linspace(0, 1, grid_points)
    pdf1 = beta.pdf(x, alpha1, beta1)
    pdf2 = beta.pdf(x, alpha2, beta2)
    return 0.5 * np.trapz(np.abs(pdf1 - pdf2), x)

@lru_cache(maxsize=1024)
def compute_prob_better(alpha1, beta1, alpha2, beta2, grid_points=50):
    """Compute probability that random variable from first beta distribution is greater than second"""
    x = np.linspace(0, 1, grid_points)
    dx = 1.0 / (grid_points - 1)
    
    # We calculate the probability in a vectorized way and use np.trapz for computational efficiency
    pdf1 = beta.pdf(x, alpha1, beta1)
    cdf2 = beta.cdf(x, alpha2, beta2)
    integrand = pdf1 * cdf2
    
    return np.trapz(integrand, x)


class EnhancedTrialEnv:
    """
    Simulated clinical trial environment with Beta–Bernoulli updates
    Enhanced with additional state features and reward shaping
    """
    
    def __init__(self, N, T,
                 init_alpha_exp, init_beta_exp,
                 init_alpha_ctrl, init_beta_ctrl,
                 p_true_exp=None, p_true_ctrl=None,
                 lambda_tvd=0.1,
                 lambda_explore=0.05,
                 seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.N = N
        self.T = T
        self.current_period = 0
        self.init_alpha_exp = init_alpha_exp
        self.init_beta_exp = init_beta_exp
        self.init_alpha_ctrl = init_alpha_ctrl
        self.init_beta_ctrl = init_beta_ctrl
        self.alpha_exp = init_alpha_exp
        self.beta_exp = init_beta_exp
        self.alpha_ctrl = init_alpha_ctrl
        self.beta_ctrl = init_beta_ctrl
        
        # True probabilities (for evaluation)
        self.p_true_exp = p_true_exp
        self.p_true_ctrl = p_true_ctrl
        
        self.lambda_tvd = lambda_tvd
        self.lambda_explore = lambda_explore
        
        # Tracks allocation history
        self.allocation_history_exp = 0
        self.allocation_history_ctrl = 0
        
        # Tracks total successes
        self.total_successes = 0
        
    def reset(self):
        self.current_period = 0
        self.alpha_exp = self.init_alpha_exp
        self.beta_exp = self.init_beta_exp
        self.alpha_ctrl = self.init_alpha_ctrl
        self.beta_ctrl = self.init_beta_ctrl
        self.alpha_exp_prev = self.init_alpha_exp
        self.alpha_ctrl_prev = self.init_alpha_ctrl
        self.allocation_history_exp = 0
        self.allocation_history_ctrl = 0
        self.total_successes = 0
        return self.get_state()
    
    def get_state(self):
        t_norm = self.current_period / self.T
        
        exp_rate = self.alpha_exp / (self.alpha_exp + self.beta_exp)
        ctrl_rate = self.alpha_ctrl / (self.alpha_ctrl + self.beta_ctrl)
        
        exp_var = (self.alpha_exp * self.beta_exp) / ((self.alpha_exp + self.beta_exp)**2 * (self.alpha_exp + self.beta_exp + 1))
        ctrl_var = (self.alpha_ctrl * self.beta_ctrl) / ((self.alpha_ctrl + self.beta_ctrl)**2 * (self.alpha_ctrl + self.beta_ctrl + 1))
        
        # Probability experimental treatment is better
        prob_exp_better = compute_prob_better(self.alpha_exp, self.beta_exp, self.alpha_ctrl, self.beta_ctrl)
        
        total_allocated = self.allocation_history_exp + self.allocation_history_ctrl
        allocation_ratio = 0.5
        if total_allocated > 0:
            allocation_ratio = self.allocation_history_exp / total_allocated
        
        return np.array([
            t_norm, 
            exp_rate, 
            ctrl_rate, 
            np.sqrt(exp_var), 
            np.sqrt(ctrl_var),
            prob_exp_better,
            allocation_ratio
        ])
    
    def step(self, action, planning_mode=False):
        if self.current_period >= self.T:
            return self.get_state(), 0.0, True
        
        tvd_old = compute_tvd(self.alpha_exp, self.beta_exp, self.alpha_ctrl, self.beta_ctrl)
        
        # We convert action to allocation
        alloc_exp = int(round(action * self.N))
        alloc_exp = max(0, min(alloc_exp, self.N))
        alloc_ctrl = self.N - alloc_exp
        
        # Update allocation history
        self.allocation_history_exp += alloc_exp
        self.allocation_history_ctrl += alloc_ctrl
        self.alpha_exp_prev = self.alpha_exp
        self.alpha_ctrl_prev = self.alpha_ctrl

        # Choose probabilities based on mode (true known vs. unknown)
        if not planning_mode and self.p_true_exp is not None and self.p_true_ctrl is not None:
            p_exp = self.p_true_exp
            p_ctrl = self.p_true_ctrl
        else:
            # Planning Mode: use current estimated probabilities
            p_exp = self.alpha_exp / (self.alpha_exp + self.beta_exp)
            p_ctrl = self.alpha_ctrl / (self.alpha_ctrl + self.beta_ctrl)
        
        # Simulate outcomes (binomial outcomes)
        successes_exp = np.random.binomial(alloc_exp, p_exp)
        successes_ctrl = np.random.binomial(alloc_ctrl, p_ctrl)
        
        # Update Beta parameters 
        self.alpha_exp += successes_exp
        self.beta_exp += (alloc_exp - successes_exp)
        self.alpha_ctrl += successes_ctrl
        self.beta_ctrl += (alloc_ctrl - successes_ctrl)

        self.total_successes += successes_exp + successes_ctrl
        
        # Reward calculation
        tvd_new = compute_tvd(self.alpha_exp, self.beta_exp, self.alpha_ctrl, self.beta_ctrl)
        immediate_reward = successes_exp + successes_ctrl
        information_gain = self.lambda_tvd * (tvd_new - tvd_old)
        
        # Exploration bonus for balanced allocation
        exploration_bonus = 0
        if alloc_exp > 0 and alloc_ctrl > 0:
            exploration_ratio = min(alloc_exp, alloc_ctrl) / max(alloc_exp, alloc_ctrl)
            # Decay exploration bonus over time
            time_decay = 1.0 - (self.current_period / self.T)
            exploration_bonus = self.lambda_explore * exploration_ratio * time_decay
        
        total_reward = immediate_reward + information_gain + exploration_bonus
        
        self.current_period += 1
        done = self.current_period >= self.T
            
        return self.get_state(), total_reward, done

    def clone(self):
        return copy.deepcopy(self)
