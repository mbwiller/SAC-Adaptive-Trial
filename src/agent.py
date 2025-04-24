# SAC Agent with automatic temperature parameter tuning using ReplayBuffer
class SACAgent:
    def __init__(self, state_dim, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005,
                 hidden_dim=128, device='cpu', alpha=0.2, lambda_aux=0, 
                 update_frequency=1, auto_alpha=True, target_entropy=None):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.lambda_aux = lambda_aux
        self.update_counter = 0
        self.update_frequency = update_frequency
        
        # Automatic temperature tuning
        self.auto_alpha = auto_alpha
        if self.auto_alpha:
            # For a bounded action space [0,1], -1.0 is a reasonable starting target entropy
            self.target_entropy = target_entropy if target_entropy is not None else -1.0
            # Initialize log_alpha as a learnable parameter
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            # Set up optimizer for alpha
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=actor_lr)
        else:
            self.alpha = alpha
        
        # Initialize networks
        self.actor = SACActorNetwork(state_dim, hidden_dim).to(self.device)
        self.critic1 = SACCriticNetwork(state_dim, hidden_dim).to(self.device)
        self.critic2 = SACCriticNetwork(state_dim, hidden_dim).to(self.device)
        self.critic1_target = SACCriticNetwork(state_dim, hidden_dim).to(self.device)
        self.critic2_target = SACCriticNetwork(state_dim, hidden_dim).to(self.device)
        
        # Initialize target networks with same weights
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Use the simplified ReplayBuffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
    
    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor.forward(state_tensor)
                action = torch.sigmoid(mean)
            else:
                action, _ = self.actor.sample(state_tensor)
                
        return action.cpu().data.numpy().flatten()[0]
    
    def update(self, batch_size=64):
        # Only update at specified frequency
        self.update_counter += 1
        if self.update_counter % self.update_frequency != 0:
            return

        # Check if enough samples are in the buffer
        if len(self.replay_buffer) < batch_size:
            return
                
        # Regular sampling from ReplayBuffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        # Use uniform weights (i.e., ones) since we're not using prioritized sampling
        weights_tensor = torch.ones(batch_size, 1).to(self.device)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Sample actions and compute Q-values for next states
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states_tensor)
            q1_next = self.critic1_target(next_states_tensor, next_actions)
            q2_next = self.critic2_target(next_states_tensor, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rewards_tensor + self.gamma * (1 - dones_tensor) * (min_q_next - self.alpha * next_log_probs)
        
        # Current Q-values
        current_q1 = self.critic1(states_tensor, actions_tensor)
        current_q2 = self.critic2(states_tensor, actions_tensor)
        
        # Critic loss using uniform weights
        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2
        critic1_loss = (weights_tensor * (td_error1 ** 2)).mean()
        critic2_loss = (weights_tensor * (td_error2 ** 2)).mean()
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Actor loss computation
        new_actions, log_probs = self.actor.sample(states_tensor)
        q1_new = self.critic1(states_tensor, new_actions)
        q2_new = self.critic2(states_tensor, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        # Standard actor loss: maximize Q - alpha * log_prob
        actor_loss = (weights_tensor * (self.alpha * log_probs - min_q_new)).mean()
        
        # Auxiliary heuristic loss based on treatment superiority
        k = 10.0  # Steepness parameter
        p_exp = states_tensor[:, 1:2]  # Experimental success rate
        p_ctrl = states_tensor[:, 2:3]  # Control success rate
        prob_exp_better = states_tensor[:, 5:6]
        heuristic_actions = torch.sigmoid(k * (p_exp - p_ctrl) + 2.0 * (prob_exp_better - 0.5))
        
        # Get deterministic actions from actor
        mean, _ = self.actor.forward(states_tensor)
        actor_deterministic = torch.sigmoid(mean)
        
        # Auxiliary loss: Mean squared error between actor's action and the heuristic
        aux_loss = F.mse_loss(actor_deterministic, heuristic_actions)
        
        # Combined actor loss
        total_actor_loss = actor_loss + self.lambda_aux * aux_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature parameter (alpha) if using automatic tuning
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            decay_factor = 0.9999  # Slow decay
            self.log_alpha.data = self.log_alpha.data * decay_factor
            self.alpha = min(self.log_alpha.exp().item(), 2.0)
        
        # Soft update target networks
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        self.actor.update_training_step()

    def plan_action(self, state, env, num_rollouts=1000):
        """
        Plan the next action based on simulated rollouts using the agent's current estimates
        
        Parameters:
          state: the current state as a numpy array
          env: the current environment instance
          num_rollouts: how many rollouts to run for each candidate action
          
        Returns:
          best_action: the action that, according to the planning simulation, appears most promising
        """
        # Prepare a tensor for the current state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        candidate_actions = []
        rewards = []
        
        # Generate a few candidate actions from the actor
        for _ in range(num_rollouts):
            # Sample candidate action (stochastic sample is fine here)
            action, _ = self.actor.sample(state_tensor)
            action_val = action.cpu().data.numpy().flatten()[0]
            candidate_actions.append(action_val)
            
            # Clone the environment so as not to perturb the true environment state
            env_clone = env.clone()
            
            # Simulate a step in planning mode: note we use planning_mode=True so that
            # the outcomes are generated from current estimates rather than true probabilities.
            _, simulated_reward, _ = env_clone.step(action_val, planning_mode=True)
            rewards.append(simulated_reward)
        
        # Select the candidate action with the highest simulated reward
        best_index = np.argmax(rewards)
        best_action = candidate_actions[best_index]
        
        return best_action

    def soft_update(self, source_net, target_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
