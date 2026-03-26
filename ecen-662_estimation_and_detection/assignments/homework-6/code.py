import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def p_gmm1(theta):
    """Target density for GMM 1: 0.5*N(-3,1) + 0.5*N(3,1)"""
    p1 = torch.exp(-0.5 * (theta + 3)**2) / (2 * torch.pi)**0.5
    p2 = torch.exp(-0.5 * (theta - 3)**2) / (2 * torch.pi)**0.5
    return 0.5 * p1 + 0.5 * p2

def p_gmm2(theta):
    """Target density for GMM 2: 0.5*N(-3,4) + 0.5*N(3,0.25)"""
    p1 = torch.exp(-0.5 * ((theta + 3)**2) / 4) / (8 * torch.pi)**0.5
    p2 = torch.exp(-0.5 * ((theta - 3)**2) / 0.25) / (0.5 * torch.pi)**0.5
    return 0.5 * p1 + 0.5 * p2

def optimize_variational_dist(target_density, init_mu, init_sigma, iterations=1000, lr=0.1):
    # Initialize parameters. We optimize log_sigma to ensure sigma stays strictly positive.
    mu = torch.tensor([float(init_mu)], requires_grad=True)
    log_sigma = torch.tensor([torch.log(torch.tensor(float(init_sigma)))], requires_grad=True)
    
    optimizer = torch.optim.Adam([mu, log_sigma], lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad()
        sigma = torch.exp(log_sigma)
        
        # 1. Re-parameterization trick
        epsilon = torch.randn(1000) # Use a batch of 1000 samples for stable expectations
        theta = mu + sigma * epsilon
        
        # 2. Calculate Loss (Negative ELBO)
        entropy = 0.5 * torch.log(2 * torch.pi * torch.e * sigma**2)
        expected_neg_log_p = -torch.mean(torch.log(target_density(theta) + 1e-8))
        
        loss = expected_neg_log_p - entropy
        
        # 3. Backpropagate and update
        loss.backward()
        optimizer.step()
        
    return mu.item(), torch.exp(log_sigma).item()

# Testing multiple initial values for GMM 1
initializations = [(-3.0, 1.0), (-3.0, 3.0), (0.0, 1.0), (0.0, 3.0), (3.0, 1.0), (3.0, 3.0)]
for init_mu, init_sigma in initializations:
    final_mu, final_sigma = optimize_variational_dist(p_gmm2, init_mu, init_sigma)
    print(f"Init: mu={init_mu:^4}, sigma={init_sigma:^4} | Final: mu={final_mu:^5.2f}, sigma={final_sigma:^5.2f}")