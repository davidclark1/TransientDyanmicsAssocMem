import torch
import numpy as np
from tqdm import tqdm


#####################################################################
######################## Regular Hopfield ###########################
#####################################################################


def simulate_hopfield(alpha, g, a, ic_noise_std, T, dt, N, N_trials=1,
    device='cpu', verbose=False):
    exclude_repeat_indices=False
    if a is not None:
        a = torch.as_tensor(a, device=device).float()
        N_condensed = len(a)
    else:
        N_condensed = 0
    
    all_m = np.zeros((N_trials, T, N_condensed)) if N_condensed > 0 else None
    all_phi = np.zeros((N_trials, T, N))
    all_x = np.zeros((N_trials, T, N))
    all_E = np.zeros((N_trials, T, 2))
    P = int(alpha * N)
    
    trial_iterator = range(N_trials)
    if verbose:
        trial_iterator = tqdm(trial_iterator, desc="Running trials")
    
    for trial in trial_iterator:

        # Patterns defining connectivity
        xi = 2 * torch.randint(0, 2, (P, N), device=device).float() - 1
        
        x = torch.zeros((T, N), device=device)
        phi = torch.zeros((T, N), device=device)
        
        if N_condensed > 0:
            # Choose initialization patterns
            init_patterns = xi[:N_condensed]
            x[0] = (init_patterns.T @ a) + ic_noise_std * torch.randn(N, device=device)
        else:
            x[0] = ic_noise_std * torch.randn(N, device=device)
        phi[0] = torch.tanh(x[0])
        
        time_iterator = range(1, T)

        for t in time_iterator:
            m_tm1 = (xi @ phi[t-1]) / N
            input = ((g / np.sqrt(alpha)) * (xi.T @ m_tm1)).clone() #- (g * np.sqrt(alpha) * phi[t-1])
            if exclude_repeat_indices:
                input -= (g * np.sqrt(alpha) * phi[t-1])
            x[t] = (1 - dt) * x[t-1] + dt * input
            phi[t] = torch.tanh(x[t])
            
            #old: all_E[trial, t-1] = compute_energy(g=g, alpha=alpha, m=m_t, x=x[t-1], phi=phi[t-1]).item()
            # Compute energy (NOTE: assumes self-connections are INCLUDED)
            all_E[trial, t-1, 0] = -(g/(2*np.sqrt(alpha)))*(m_tm1**2).sum() 
            all_E[trial, t-1, 1] =  compute_F(x[t-1], phi[t-1]) 

        m_last = (xi @ phi[-1]) / N
        all_E[trial, -1, 0] = -(g/(2*np.sqrt(alpha)))*(m_last**2).sum() 
        all_E[trial, -1, 1] =  compute_F(x[-1], phi[-1]) 

        #old: all_E[trial, -1] = compute_energy(g=g, alpha=alpha, m=(xi @ phi[-1]) / N, x=x[-1], phi=phi[-1], 
        #                                  exclude_repeat_indices=exclude_repeat_indices).item()

        all_phi[trial] = phi.cpu().numpy()
        all_x[trial] = x.cpu().numpy()
        if N_condensed > 0:
            all_m[trial] = ((phi @ init_patterns.T) / N).cpu().numpy()
            
    return all_m, all_x, all_phi, all_E

def compute_energy(g, alpha, m, x, phi, exclude_repeat_indices=False):
    N = phi.shape[0]
    interaction_term = -(g*N / (2*np.sqrt(alpha))) * torch.sum(m**2)
    crosstalk_term = ((g*np.sqrt(alpha) / 2) * torch.sum(phi**2)) if exclude_repeat_indices else 0.
    single_neuron_term = N*compute_energy_single_neuron_term(x, phi)
    
    E = (interaction_term + crosstalk_term + single_neuron_term) / N
    return E

def compute_F(x, phi): #SINGLE NEURON ENERGY term
    # GIVES: F(\phi) = \frac{1}{2}\log\left(1 - \phi^2\right) + x \phi (can check equivalence)
    #x, phi = (T, M) or (M,)
    if x.ndim == 1:
        to_item = True
        x, phi = x[None, :], phi[None, :]
    else:
        to_item = False
    #print(x.std())
    single_neuron_term = np.log(2) + torch.mean(x - torch.nn.functional.softplus(2*x, threshold=50.) + x*phi, dim=1)
    if to_item:
        return single_neuron_term.item()
    return single_neuron_term

def solve_hopfield_dmft(alpha, g, a, ic_noise_std, T, dt, M, N_iter, update_stepsize, device='cpu', verbose=False):
    # Infer N_condensed from a
    N_condensed = 0 if a is None or len(a) == 0 else len(a)
    
    # Initialize parameters
    if N_condensed > 0:
        a = torch.as_tensor(a, device=device).type(torch.float)
        m = torch.randn((T, N_condensed), device=device) * 0.1
    else:
        m = None
        
    C_phi = torch.eye(T, device=device)
    S_phi = torch.zeros((T, T), device=device)
    E = torch.zeros((T, 2), device=device)

    # Helper functions
    nonlin = torch.tanh
    nonlin_deriv = lambda x: 1 - torch.tanh(x)**2

    epsilon_schedule = np.ones(N_iter)
    epsilon_schedule[:20] = np.linspace(0, 1, 20)**4

    trial_iterator = range(N_iter)
    if verbose:
        trial_iterator = tqdm(trial_iterator, desc="solving for order parameters")
    
    for iter_idx in trial_iterator:
        # Compute K and related quantities
        m_response_inv = torch.eye(T, device=device) - epsilon_schedule[iter_idx] * (g/np.sqrt(alpha)) * S_phi
        m_response = torch.linalg.solve_triangular(m_response_inv, torch.eye(T, device=device), upper=False, unitriangular=True)
        
        if torch.any(torch.isnan(m_response)) or torch.any(torch.isinf(m_response)):
            if verbose: print("m_response is bad (presumably bad inversion)")
            return
        
        # Sample Gaussian noise
        C_eta = (g**2) * m_response @ C_phi @ m_response.T
        L_npy = np.linalg.cholesky(C_eta.cpu().numpy() + np.eye(T)*(1e-3 * (g/1.5)**2))
        L = torch.Tensor(L_npy).to(device)
        eta = L @ torch.randn(T, M, device=device)

        # Define self-coupling kernel
        F = g * np.sqrt(alpha) * m_response

        # Initialize timeseries
        x = torch.zeros((T, M), device=device)
        phi = torch.zeros((T, M), device=device)
        
        # Handle condensed patterns if N_condensed > 0
        if N_condensed > 0:
            xi = 2 * torch.randint(0, 2, (M, N_condensed), device=device).float() - 1
            x[0] = xi @ a + ic_noise_std * torch.randn(M, device=device)
        else:
            x[0] = ic_noise_std * torch.randn(M, device=device)
            
        phi[0] = nonlin(x[0])

        # Run Euler integration
        for t in range(1, T):
            input = eta[t-1] + (F[t-1, 0:t] @ phi[0:t])
            if N_condensed > 0:
                input += (g/np.sqrt(alpha)) * (xi @ m[t-1])
            x[t] = (1-dt)*x[t-1] + dt*input
            phi[t] = nonlin(x[t])

        # Compute new order parameters
        C_phi_new = (phi @ phi.T) / M
        if N_condensed > 0:
            m_new = (phi @ xi) / M

        # Integrate response functions
        S_x_new = torch.zeros((T, T, M), device=device)
        phi_prime = nonlin_deriv(x)
        
        for s in range(T):
            for t in range(s, T):
                if t == s:
                    S_x_new[t,s] = 0
                elif t == s+1:
                    S_x_new[t,s] = dt
                else:
                    input = (F[t-1,s:t] @ (phi_prime[s:t] * S_x_new[s:t,s]))
                    S_x_new[t,s] = (1-dt)*S_x_new[t-1,s] + dt*input

        S_phi_new = phi_prime.unsqueeze(1) * S_x_new
        
        # Average over trajectories
        S_phi_new_tot = S_phi_new.mean(-1)
        
        # Update order parameters with memory
        S_phi = update_stepsize*S_phi_new_tot + (1 - update_stepsize)*S_phi
        C_phi = update_stepsize*C_phi_new + (1 - update_stepsize)*C_phi
        if N_condensed > 0:
            m = update_stepsize*m_new + (1 - update_stepsize)*m
        
        # Compute energy if last iter
        if iter_idx == N_iter - 1:
            term_1 = -(np.sqrt(alpha)/(2*g))*torch.diag(C_eta) - (g/(2*np.sqrt(alpha)))*(m**2).sum(1)
            term_2 = compute_F(x, phi) #x,phi = (T,M)
            E[:, 0] = term_1
            E[:, 1] = term_2
            
    if m is not None:
        m = m.cpu().numpy()
        
    return m, C_phi.cpu().numpy(), S_phi.cpu().numpy(), E.cpu().numpy()


#####################################################################
################### Dense associative memory ########################
#####################################################################

def simulate_dense_assoc_mem(alpha, n, g, a, ic_noise_std, T, dt, N, N_trials=1, device='cpu',
    verbose=False, exclude_repeat_indices=False):
    if a is not None:
        a = torch.as_tensor(a, device=device).float()
        N_condensed = len(a)
    else:
        N_condensed = 0
    
    all_m = np.zeros((N_trials, T, N_condensed)) if N_condensed > 0 else None
    all_phi = np.zeros((N_trials, T, N))
    all_x = np.zeros((N_trials, T, N))
    all_E = np.zeros((N_trials, T, 2)) #2 components
    P = int(alpha * (N**n))
    if verbose: print("P =", P)
    
    trial_iterator = range(N_trials)
    if verbose:
        trial_iterator = tqdm(trial_iterator, desc="Running trials")
    
    for trial in trial_iterator:
        
        # Patterns defining connectivity
        xi = 2 * torch.randint(0, 2, (P, N), device=device).float() - 1
        
        x = torch.zeros((T, N), device=device)
        phi = torch.zeros((T, N), device=device)
        
        if N_condensed > 0:
            # Choose initialization patterns
            init_patterns = xi[:N_condensed]
            x[0] = (init_patterns.T @ a) + ic_noise_std * torch.randn(N, device=device)
        else:
            x[0] = ic_noise_std * torch.randn(N, device=device)
        phi[0] = torch.tanh(x[0])
        
        time_iterator = range(1, T)
        for t in time_iterator:
            m_tm1 = (xi @ phi[t-1]) / N
            f_tm1 = m_tm1**n
            input = (g / np.sqrt(alpha)) * (xi.T @ f_tm1)
            if exclude_repeat_indices:
                #TODO: for n=2 and n=3 cases, implement formulas from latex notes (subtract from input)
                # if n is neither 2 nor 3, raise error
                pass
            x[t] = (1 - dt) * x[t-1] + dt * input
            phi[t] = torch.tanh(x[t])

            all_E[trial, t-1, 0] = -(g/((n+1)*np.sqrt(alpha)))*(m_tm1**(n+1)).sum() 
            all_E[trial, t-1, 1] =  compute_F(x[t-1], phi[t-1]) 

        m_last = (xi @ phi[-1]) / N
        all_E[trial, -1, 0] = -(g/((n+1)*np.sqrt(alpha)))*(m_last**(n+1)).sum() 
        all_E[trial, -1, 1] =  compute_F(x[-1], phi[-1]) 

        all_phi[trial] = phi.cpu().numpy()
        all_x[trial] = x.cpu().numpy()
        if N_condensed > 0:
            all_m[trial] = ((phi @ init_patterns.T) / N).cpu().numpy()
            
    return all_m, all_x, all_phi, all_E


def solve_dense_assoc_mem_dmft(g, alpha, n, a, ic_noise_std, T, dt, M, N_iter, update_stepsize, device='cpu', verbose=False):
    # Infer N_condensed from a
    N_condensed = 0 if a is None or len(a) == 0 else len(a)
    
    # Initialize parameters
    if N_condensed > 0:
        a = torch.as_tensor(a, device=device).type(torch.float)
        m = torch.ones((T, N_condensed), device=device)
    else:
        m = None
        
    C_phi = torch.eye(T, device=device) / np.sqrt(dt)
    S_phi = torch.zeros((T,T), device=device)
    E = torch.zeros((T, 2), device=device)

    # Helper functions
    nonlin = torch.tanh
    nonlin_deriv = lambda x: 1 - torch.tanh(x)**2

    trial_iterator = range(N_iter)
    if verbose:
        trial_iterator = tqdm(trial_iterator, desc="solving for order parameters")

    thing = 0.
    for iter_idx in trial_iterator:
        on_diag = torch.diag(C_phi)
        if n == 2:
            P_n_nm2 = torch.outer(on_diag, torch.ones_like(on_diag))
            P_nm1_nm1 = C_phi
            P_n_n = torch.outer(on_diag, on_diag) + 2*(C_phi**2)
        elif n == 4:
            P_n_nm2 = 3*torch.outer(on_diag**2, on_diag) + 12*(on_diag[:, None] * C_phi**2)
            P_nm1_nm1 = 9*torch.outer(on_diag, on_diag)*C_phi + 6*C_phi**3
            P_n_n = 9*torch.outer(on_diag, on_diag)**2 + 72*torch.outer(on_diag, on_diag)*C_phi**2 + 24*C_phi**4
        else:
            raise ValueError("n must be 2 or 4")
        F = (g**2) * ((n*(n-1) * torch.diag(torch.diag(S_phi @ P_n_nm2))) + ((n**2) * (S_phi * P_nm1_nm1)))
        C_eta = (g**2) * P_n_n

        ## Sample Gaussian noise
        #(do the Cholesky in numpy due to weird CUDA error)
        L_npy = np.linalg.cholesky(C_eta.cpu().numpy() + np.eye(T)*(1e-3 * (g/1.5)**2))
        L = torch.Tensor(L_npy).to(device)
        eta = L @ torch.randn(T, M, device=device)
        
        # Initialize timeseries
        x = torch.zeros((T, M), device=device)
        phi = torch.zeros((T, M), device=device)
        
        # Handle condensed patterns if N_condensed > 0
        if N_condensed > 0:
            xi = 2 * torch.randint(0, 2, (M, N_condensed), device=device).float() - 1
            x[0] = xi @ a + ic_noise_std * torch.randn(M, device=device)
        else:
            x[0] = ic_noise_std * torch.randn(M, device=device)
            
        phi[0] = nonlin(x[0])

        # Run Euler integration
        for t in range(1, T):
            input = eta[t-1] + (F[t-1, 0:t, None] * phi[0:t, :]).sum(0)
            if N_condensed > 0:
                input += (g / np.sqrt(alpha)) * (xi * (m[t-1, None, :]**n)).sum(1)
            x[t] = (1-dt)*x[t-1] + dt*input
            phi[t] = nonlin(x[t])

        # Compute new order parameters
        C_phi_new = (phi @ phi.T) / M
        if N_condensed > 0:
            m_new = (phi @ xi) / M

        # Integrate response functions
        S_x_new = torch.zeros((T, T, M), device=device)
        phi_prime = nonlin_deriv(x)
        
        for s in range(T):
            for t in range(s, T):
                if t == s:
                    S_x_new[t,s] = 0. 
                elif t == s+1:
                    S_x_new[t,s] = dt
                else:
                    input = (F[t-1,s:t,None] * phi_prime[s:t,:] *  S_x_new[s:t,s,:]).sum(0)
                    S_x_new[t,s] = (1-dt)*S_x_new[t-1,s] + dt*input

        S_phi_new = phi_prime[:, None, :] * S_x_new
        
        # Average over trajectories
        S_phi_new_tot = S_phi_new.mean(-1)
        
        # Update order parameters with memory
        S_phi = update_stepsize*S_phi_new_tot + (1 - update_stepsize)*S_phi
        C_phi = update_stepsize*C_phi_new + (1 - update_stepsize)*C_phi
        if N_condensed > 0:
            m = update_stepsize*m_new + (1 - update_stepsize)*m
            
        # Compute energy
        energy_1_new = -torch.mean(phi * eta, dim=1) - (g/((n+1)*np.sqrt(alpha)))*(m**(n+1)).sum(1)
        energy_2_new = compute_F(x, phi)
        E[:, 0] = update_stepsize*energy_1_new + (1-update_stepsize)*energy_1_new
        E[:, 1] = update_stepsize*energy_2_new + (1-update_stepsize)*energy_2_new

    if m is not None:
        m = m.cpu().numpy()
        
    return m, C_phi.cpu().numpy(), S_phi.cpu().numpy(), E.cpu().numpy()

