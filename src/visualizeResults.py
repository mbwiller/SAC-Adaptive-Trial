def visualize_results(results_df, save_prefix=None):
    """
    Create comprehensive visualizations for the parameter study results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing parameter study results
    save_prefix : str or None
        Prefix for saving visualization files, if None will display instead
    """
    if save_prefix:
        os.makedirs("results/plots_vfinal_22F", exist_ok=True)
        save_path = f"results/plots_vfinal_22F/{save_prefix}"
    else:
        save_path = "temp_plot_not_sac"
    
    plt.figure(figsize=(15, 10))
    num_rows = (len(results_df) + 3) // 4
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        plt.subplot(num_rows, 4, i+1)
        
        values = []
        labels = []
        colors = []
        
        for method, color in [('fixed', 'gray'), ('greedy', 'blue'), 
                             ('pl', 'green'), ('sac', 'purple')]:
            if f'{method}_success' in row:
                val = row[f'{method}_success']
                values.append(val)
                labels.append(method.upper())
                colors.append(color)
        
        ideal = row['ideal_success'] if 'ideal_success' in row else None
        
        title = f"Scenario {i+1}"
        if 'scenario' in row:
            title = row['scenario']
            
        if 'N' in row and 'T' in row:
            title += f"\nN={row['N']}, T={row['T']}"
        
        bars = plt.bar(labels, values, color=colors)
        
        if ideal is not None:
            plt.axhline(y=ideal, color='red', linestyle='-', alpha=0.7, label='Ideal')
            plt.legend()
        
        plt.title(title)
        plt.ylim(0, max(values + [ideal if ideal is not None else 0]) * 1.1)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_path}_success_rates.png")
        plt.close()
    else:
        plt.show()
    
    plt.figure(figsize=(15, 10))
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        plt.subplot(num_rows, 4, i+1)
        
        allocs = []
        labels = []
        colors = []
        
        for method, color in [('fixed', 'gray'), ('greedy', 'blue'), 
                             ('pl', 'green'), ('sac', 'purple')]:
            if f'{method}_alloc' in row:
                allocs.append(row[f'{method}_alloc'])
                labels.append(method.upper())
                colors.append(color)
        
        if 'ideal_alloc' in row:
            allocs.append(row['ideal_alloc'])
            labels.append('IDEAL')
            colors.append('red')
        
        if allocs:
            bars = plt.bar(labels, allocs, color=colors)
            
            plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height,
                        f'{height:.2f}', ha='center', va='bottom')
            
            title = f"Scenario {i+1}: Allocation"
            if 'scenario' in row:
                title = f"{row['scenario']}: Allocation"
            
            plt.title(title)
            plt.ylim(0, 1.1)
    
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_path}_allocations.png")
        plt.close()
    else:
        plt.show()

    
    if 'p_diff' in results_df.columns and 'sac_success' in results_df.columns:
        plt.figure(figsize=(12, 8))
        
        p_diff = results_df['p_diff']
        abs_diff = np.abs(p_diff)
        
        sizes = results_df['N'] * results_df['T'] 
        sizes = (sizes / max(sizes)) * 300 + 50
        
        scatter = plt.scatter(p_diff, results_df['sac_success'], c=abs_diff, cmap='viridis', 
                            s=sizes, alpha=0.7, edgecolors='black')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('|p_exp - p_ctrl|')
        
        if len(p_diff) > 1:
            z = np.polyfit(p_diff, results_df['sac_success'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(p_diff), max(p_diff), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.7)
        
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        for i, (pd, perf) in enumerate(zip(p_diff, results_df['sac_success'])):
            label = str(i+1)
            if 'scenario' in results_df.columns:
                label = results_df.iloc[i]['scenario'].split('-')[0].strip()
            plt.annotate(label, (pd, perf), fontsize=9)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('p_true_exp - p_true_ctrl')
        plt.ylabel('SAC Success Proportion')
        plt.title('Effect of True Probability Difference on SAC Performance')
        
        if save_prefix:
            plt.savefig(f"{save_path}_p_diff_effect.png")
            plt.close()
        else:
            plt.show()

def plot_sac_trajectory(trajectories, env_params, save_prefix=None, scenario_name=None, scenario_idx=0, timestamp=""):
    if save_prefix:
        os.makedirs("results/plots_vfinal_22F", exist_ok=True)
        save_path = f"results/plots_vfinal_22F/{save_prefix}"
    else:
        save_path = "temp_plot_sac"

    steps = [t['period'] for t in trajectories]
    
    allocations = [
        t['allocation_exp'] / (t['allocation_exp'] + t['allocation_ctrl'])
        if (t['allocation_exp'] + t['allocation_ctrl']) > 0 else 0.5 
        for t in trajectories
    ]
    
    exp_beliefs = [t['exp_rate'] for t in trajectories]
    ctrl_beliefs = [t['ctrl_rate'] for t in trajectories]
    
    belief_diff = [e - c for e, c in zip(exp_beliefs, ctrl_beliefs)]
    
    total_uncertainty = []
    for t in trajectories:
        if all(k in t for k in ['alpha_exp', 'beta_exp', 'alpha_ctrl', 'beta_ctrl']):
            print("using updated parameters for trajectory plotting")
            a_exp, b_exp = t['alpha_exp'], t['beta_exp']
            a_ctrl, b_ctrl = t['alpha_ctrl'], t['beta_ctrl']
        else:
            print("falling back to initial values for trajectory plotting")
            a_exp, b_exp = env_params['init_alpha_exp'], env_params['init_beta_exp']
            a_ctrl, b_ctrl = env_params['init_alpha_ctrl'], env_params['init_beta_ctrl']
        
        # Compute standard deviation of a Beta(α, β) distribution:
        # std = sqrt((α * β) / ((α+β)^2 * (α+β+1)))
        exp_std = np.sqrt((a_exp * b_exp) / ((a_exp + b_exp)**2 * (a_exp + b_exp + 1)))
        ctrl_std = np.sqrt((a_ctrl * b_ctrl) / ((a_ctrl + b_ctrl)**2 * (a_ctrl + b_ctrl + 1)))
        
        # Combine uncertainties for the difference (assuming independence)
        combined_std = np.sqrt(exp_std**2 + ctrl_std**2)
        total_uncertainty.append(combined_std)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(steps, allocations, 'o-', color='purple', label='SAC')
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50-50 Split')
    optimal_alloc = 1.0 if env_params['p_true_exp'] > env_params['p_true_ctrl'] else 0.0
    plt.axhline(y=optimal_alloc, color='red', linestyle='-.', alpha=0.5, label='Oracle Optimal')
    plt.title("SAC Allocation Trajectory")
    plt.ylabel("Proportion to Experimental Arm")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(steps, exp_beliefs, 'r-', label='Exp. Rate Belief')
    plt.plot(steps, ctrl_beliefs, 'b-', label='Ctrl. Rate Belief')
    plt.axhline(y=env_params['p_true_exp'], color='r', linestyle='--', alpha=0.5, label='True Exp. Rate')
    plt.axhline(y=env_params['p_true_ctrl'], color='b', linestyle='--', alpha=0.5, label='True Ctrl. Rate')
    plt.title("Belief Trajectories")
    plt.ylabel("Success Rate Belief")
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(steps, belief_diff, 'g-', label='Belief Difference (Exp - Ctrl)')
    # Lower bound = belief_diff - combined uncertainty; upper bound = belief_diff + combined uncertainty
    lower_bound = [d - u for d, u in zip(belief_diff, total_uncertainty)]
    upper_bound = [d + u for d, u in zip(belief_diff, total_uncertainty)]
    plt.fill_between(steps, lower_bound, upper_bound, color='g', alpha=0.2, label='Uncertainty Bounds')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    true_diff = env_params['p_true_exp'] - env_params['p_true_ctrl']
    plt.axhline(y=true_diff, color='purple', linestyle='-.', alpha=0.5, label=f'True Difference ({true_diff:.3f})')
    plt.title("Belief Difference and Uncertainty")
    plt.xlabel("Time Step")
    plt.ylabel("Belief Difference")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    suptitle = "SAC Trajectory Analysis"
    if scenario_name:
        suptitle += f" - {scenario_name}"
    plt.suptitle(suptitle, fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    if save_prefix:
        plot_filename = f"{save_path}_sac_trajectory.png"
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        return plot_filename
    else:
        plt.show()
        return None
