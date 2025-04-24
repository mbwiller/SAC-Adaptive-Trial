# Bayesian Adaptive Clinical Trials via Soft Actor-Critic

This repository contains the code and supporting materials for **Bayesian Adaptive Clinical Trials: A Soft Actor-Critic Reinforcement Learning Approach**,  
Matthew Willer (Princeton University, May 2025).

Adaptive clinical trial designs dynamically allocate patients based on accruing data to improve efficiency and ethics.

We model two-arm trials as a finite-horizon MDP with Beta–Bernoulli updates and train a Soft Actor-Critic agent to optimize patient allocations.

A Total Variation Distance term quantifies information gain, balancing exploration and exploitation across diverse scenarios.

_(Note: ChatGPT was used to help with coding)_
