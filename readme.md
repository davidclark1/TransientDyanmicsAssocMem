# Transient Dynamics of Associative Memory Models

This repository contains code for solving dynamical mean field theory (DMFT) equations for Hopfield and dense associative memory models, as described in:

**"Transient dynamics of associative memory models"** by David G. Clark  
Paper: https://arxiv.org/abs/2506.05303

## Overview

The code implements both simulation and theoretical DMFT solutions for:
- **Hopfield networks**: Classical associative memory with pairwise interactions
- **Dense associative memory**: Generalized model with higher-order ((n+1)-th order) interactions

Both models support:
- Direct numerical simulation of finite-size network dynamics
- Theoretical DMFT solution of self-consistent equations
- Energy calculation

## Usage

All functionality is contained in `dmft.py`. Import the module and required dependencies:

```python
import dmft
import numpy as np
import torch
```

### Hopfield Network

```python
# Model parameters
alpha = 0.1             # Pattern loading (P = alpha * N stored patterns)
g = 1.2                 # Coupling strength
a_scale = 0.9           # Initial pattern overlap strength
a = np.array([a_scale * g])  # Initial pattern weights
ic_noise_std = np.sqrt(max(0, g**2 - a[0]**2))  # Initial condition noise

# Simulation parameters
T = 100                 # Number of time steps
dt = 0.02              # Time step size
N = 200                # Network size
N_trials = 20          # Number of simulation trials
device = 'cpu'         # Device

# Run direct simulation
m_sims, x_sims, phi_sims, E_sims = dmft.simulate_hopfield(
    alpha=alpha,
    g=g,
    a=a,
    ic_noise_std=ic_noise_std,
    T=T,
    dt=dt,
    N=N,
    N_trials=N_trials,
    device=device,
    verbose=True
)

# DMFT solution parameters
M = 50000              # Number of sample trajectories
N_iter = 150           # Number of iterations
update_stepsize = 0.3  # Learning rate

# Solve DMFT equations
m_dmft, C_phi_dmft, S_phi_dmft, E_dmft = dmft.solve_hopfield_dmft(
    alpha=alpha,
    g=g,
    a=a,
    ic_noise_std=ic_noise_std,
    T=T,
    dt=dt,
    M=M,
    N_iter=N_iter,
    update_stepsize=update_stepsize,
    device=device,
    verbose=True
)
```

### Dense Associative Memory

```python
# Model parameters
n = 2                   # Interaction order, can be n=2 or n=4
alpha = 0.05            # Pattern loading parameter (P = alpha * N^n stored patterns)
g = 1.5                 # Coupling strength
a_scale = 0.8           # Initial pattern overlap strength
a = np.array([a_scale * g])  # Initial pattern weights
ic_noise_std = np.sqrt(g**2 - a[0]**2)  # Initial condition noise level

# Simulation parameters
T = 50                  # Number of time steps
dt = 0.05              # Time step size
N = 100                # Network size
N_trials = 50          # Number of simulation trials
device = 'cpu'         # Device ('cpu' or 'cuda')

# Run direct simulation
m_sims, x_sims, phi_sims, E_sims = dmft.simulate_dense_assoc_mem(
    n=n,
    alpha=alpha,
    g=g,
    a=a,
    ic_noise_std=ic_noise_std,
    T=T,
    dt=dt,
    N=N,
    N_trials=N_trials,
    device=device,
    verbose=True
)

# DMFT solution parameters
M = 100000             # Number of sample trajectories for DMFT
N_iter = 200           # Number of DMFT iterations
update_stepsize = 0.5  # Learning rate for order parameter updates

# Solve DMFT equations
m_dmft, C_phi_dmft, S_phi_dmft, E_dmft = dmft.solve_dense_assoc_mem_dmft(
    g=g,
    alpha=alpha,
    n=n,
    a=a,
    ic_noise_std=ic_noise_std,
    T=T,
    dt=dt,
    M=M,
    N_iter=N_iter,
    update_stepsize=update_stepsize,
    device=device,
    verbose=True
)
```

## Function Parameters

### Simulation Functions

**`simulate_hopfield()` / `simulate_dense_assoc_mem()`**
- `alpha`: Pattern loading parameter
- `g`: Coupling strength
- `a`: Initial pattern overlap weights (array)
- `ic_noise_std`: Standard deviation of initial condition noise
- `T`: Number of time steps
- `dt`: Time step size
- `N`: Network size
- `N_trials`: Number of independent simulation trials
- `device`: Compute device ('cpu' or 'cuda')

### DMFT Solution Functions

**`solve_hopfield_dmft()` / `solve_dense_assoc_mem_dmft()`**
- `M`: Number of sample trajectories for DMFT averaging
- `N_iter`: Number of self-consistency iterations
- `update_stepsize`: Learning rate for order parameter updates
- Additional parameters same as simulation functions

### Dense Associative Memory Specific
- `n`: Interaction order

## Output Variables

### Simulation Outputs
- `m_sims`: Pattern overlaps over time (trials × time × patterns)
- `x_sims`: Neuron states (trials × time × neurons)  
- `phi_sims`: Neuron activations (tanh of states) (trials × time × neurons)
- `E_sims`: Energy components (trials × time × 2)

### DMFT Outputs
- `m_dmft`: Theoretical pattern overlaps (time × patterns)
- `C_phi_dmft`: Activity correlation matrix (time × time)
- `S_phi_dmft`: Response function matrix (time × time)
- `E_dmft`: Theoretical energy components (time × 2)
