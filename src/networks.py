class SACActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(SACActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Maintaining same width
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # Output layers
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)
        
        # Adaptive log std bounds
        self.log_std_min = -20.0
        self.log_std_max = 2.0
        self.training_step = 0
        self.max_training_steps = 100000
    
    def forward(self, state):
        x1 = F.elu(self.ln1(self.fc1(state)))
        x2_pre = self.ln2(self.fc2(x1))
        x2 = F.elu(x2_pre) + x1  # Residual connection
        x3_pre = self.ln3(self.fc3(x2))
        x3 = F.elu(x3_pre) + x2  # Residual connection
        
        mean = self.mean(x3)
        log_std = self.log_std(x3)
        
        # Calculate adaptive bounds based on training progress
        progress = min(1.0, self.training_step / self.max_training_steps)
        current_min = self.log_std_min
        current_max = self.log_std_max - progress * 1.0
        
        log_std = torch.clamp(log_std, current_min, current_max)
        
        return mean, log_std
    
    def update_training_step(self, step_increment=1):
        self.training_step += step_increment
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        z = normal.rsample()
        
        # Apply sigmoid to constrain actions to [0, 1]
        action = torch.sigmoid(z)
        
        # Calculate log probability, correcting for the transform
        log_prob = normal.log_prob(z) - torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

class SACCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(SACCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + 1, hidden_dim)  # +1 for the action
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x1 = F.elu(self.ln1(self.fc1(x)))
        x2_pre = self.ln2(self.fc2(x1))
        x2 = F.elu(x2_pre) + x1  # Residual connection
        x3_pre = self.ln3(self.fc3(x2))
        x3 = F.elu(x3_pre) + x2  # Residual connection
        q_value = self.out(x3)
        return q_value
