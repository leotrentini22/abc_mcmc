import numpy as np
import scipy.stats as st
import random
import math
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import *

def discrepancy_metric(x, sample_mean=0):
    '''
    Computation of discrepancy metric p = |mean(x)-sample_mean|.
    '''
    # rho(S(D*),S(D)) = abs(mean(x*) - mean(x)), we assume that mean(x)=0 (sample mean)
    return abs(np.mean(x) - sample_mean)


def true_mixture_distribution(x, M, a, var, var_1, sample_mean=0):
    '''
    Computation of true mixture distribution of the academic example.
    '''
    alfa = 1./( 1 + np.exp( a * (sample_mean - 0.5*a) * M / (M*var + var_1) ) )
    pdf_1 = alfa * st.norm.pdf(x=x, loc = var/(var+var_1/M)*sample_mean, scale = np.sqrt(var_1/(M+var_1/var)))
    pdf_2 = (1-alfa) * st.norm.pdf(x=x, loc = var/(var+var_1/M)*(sample_mean-a), scale = np.sqrt(var_1/(M+var_1/var)))
    # f = alfa * st.norm(loc = var/(var+var_1/M)*sample_mean, scale = var_1/(M+var_1/var)) + (1-alfa) * st.norm(loc = var/(var+var_1/M)*(sample_mean-a), scale = var_1/(M+var_1/var))
    f = pdf_1 + pdf_2
    return f


def basic_abc(N, M, eps, var, var_1, a):
    '''
    Implementation of basic ABC algorithm for the academic example.
    N = number of iterations
    M = data for each iteration
    eps = tolerance
    var, var_1, a = data related to the academic example
    '''
    theta = np.zeros(N) #at the end of the algorithm it has to be full (algorithm ends when we have selected N samples)
    pi_rv = st.norm(loc=0, scale=np.sqrt(var))
    rejection_count = 0
    n1_count = 0  #DEBUG
    n2_count = 0  #DEBUG
    i = 0 # count of the number of selected samples
    
    while i < N:
        # sample candidate parameters from the prior distribution pi = N(0,var)
        theta_star = pi_rv.rvs()
    
        # generate data from the underlying model given theta_star
        # D observed data, is an iid sample drawn with prob=0.5 from N(theta,var_1), o/w from N(theta+a,var_1)
        N1_rv = st.norm(loc=theta_star, scale=np.sqrt(var_1))
        N2_rv = st.norm(loc=theta_star + a, scale=np.sqrt(var_1))
        D = np.zeros(M)
        
        #build a set of M observation from the underlying model given theta_star
        if np.random.choice([0, 1]) == 0:
            for j in range(M):
                D[j] = N1_rv.rvs()
                n1_count += 1
        else:
            for j in range(M):
                D[j] = N2_rv.rvs()
                n2_count += 1
        
        if discrepancy_metric(D) < eps:
            theta[i] = theta_star
            i += 1 # a new sample is selected
        else:
            rejection_count += 1
    
    print('acceptance rate:', N/(N + rejection_count))
    return theta, N/(N + rejection_count)


def abc_mcmc(N_max, M, eps, nu_squared, var, var_1, a):
    '''
    Implementation of ABC MCMC algorithm for the academic example.
    N = number of iterations
    M = data for each iteration
    eps = tolerance
    nu_squared = variance of (Gaussian) transition probability
    var, var_1, a = data related to the academic example
    '''
    theta = np.zeros(N_max+1)
    pi_rv = st.norm(loc=0, scale=np.sqrt(nu_squared))  # prior 
    
    acceptance_count = 0
    rejection_count = 0
    entered = 0
    i = 0
    metropoli_ratio_list = []
    
    while i < N_max:
        # sample candidate parameters from a proposal transition density q(theta_i, )
        # random walk proposal q(theta, ) = N(theta,nu_squared)
        q_i = st.norm(loc=theta[i], scale=np.sqrt(nu_squared))
        theta_star = q_i.rvs()
    
        # generate data from the underlying model given theta_star
        N1_rv = st.norm(loc=theta_star, scale=np.sqrt(var_1))
        N2_rv = st.norm(loc=theta_star + a, scale=np.sqrt(var_1))
        
        D = np.zeros(M)
        if np.random.choice([0, 1]) == 0:
            for j in range(M):
                D[j] = N1_rv.rvs()
                # n1_count = n1_count + 1
        else:
            for j in range(M):
                D[j] = N2_rv.rvs()
                # n2_count = n1_count + 1
    
        if discrepancy_metric(D) < eps:
            q_star = st.norm(loc=theta_star, scale=np.sqrt(nu_squared))
            comp = pi_rv.pdf(theta_star)*q_star.pdf(theta[i])/( pi_rv.pdf(theta[i])*q_i.pdf(theta_star) )
            metropoli_ratio_list.append(min(1.,comp))
            
            entered += 1
            
            if st.uniform.rvs() < comp:
                theta[i+1] = theta_star
                acceptance_count += 1 
            else:
                theta[i+1] = theta[i]
                rejection_count += 1
        else:
            theta[i+1] = theta[i]
            rejection_count += 1
            
        i += 1
    
    print('acceptance rate:', acceptance_count/(acceptance_count+rejection_count))
    return theta, acceptance_count/(acceptance_count+rejection_count)


def simulate_Xt_times(K_e, K_a, Cl, sigma):
    '''
    Simulation of the pharmacokinetics model with Euler-Maruyama and selection of the 9 times we are interested in.
    Return only the value of the process at those times.
    '''
    dt = 0.05  # Time step
    T = 12  # Total time
    n = int(T / dt)+1  # Number of time steps
    D = 4

    t = np.arange(0, 12.05, 0.05)  # Vector of times

    X_t = np.zeros(n) # Recall that X_0 = 0
    for i in range(n - 1):
        X_t[i + 1] = X_t[i] + dt*( (D*K_a*K_e)/Cl * np.exp(-K_a*t[i+1]) - K_e*X_t[i] ) + sigma*np.sqrt(dt)*np.random.randn()
    
    times = [0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12]
    indices = np.where(np.isin(t, times))
    Data = X_t[indices]
    return Data


def simulate_Xt_full(K_e, K_a, Cl, sigma):
    '''
    Simulation of the pharmacokinetics model with Euler-Maruyama.
    Return all the times and all the values of the stochastic process X_t for each time evaluated.
    '''
    dt = 0.05  # Time step
    T = 12  # Total time
    n = int(T / dt)+1  # Number of time steps
    D = 4

    t = np.arange(0, 12.05, 0.05)  # Vector of times

    X_t = np.zeros(n) # Recall that X_0 = 0
    for i in range(n - 1):
        X_t[i + 1] = X_t[i] + dt*( (D*K_a*K_e)/Cl * np.exp(-K_a*t[i+1]) - K_e*X_t[i] ) + sigma*np.sqrt(dt)*np.random.randn()
    
    return t, X_t


def new_discrepancy_metric(D_star, S_D, intercept, coefficients, theta_0):
    '''
    Computation of the discrepancy metric p = ||S(D)-S(D*)||.
    '''
    S_D_star = intercept + coefficients@D_star
    
    diff = S_D_star - S_D
    metric = 0
    
    for i in range(len(diff)):
        metric += diff[i]**2/(theta_0[i])**2
        
    return metric


def jointly_prior_sample(prior_K_e, prior_K_a, prior_Cl, prior_sigma, theta):
    '''
    Return the jointly prior evaluated in theta assuming independence.
    '''
    # assuming independence
    return prior_K_e.pdf(theta[0])*prior_K_a.pdf(theta[1])*prior_Cl.pdf(theta[2])*prior_sigma.pdf(theta[3])


def abc_mcmc_pharma(N_max, M, eps, S_D, intercept, coefficients):
    '''
    Implementation of ABC-MCMC algorithm for the pharmacokinetics model.
    N_max = maximum number of iterations
    M = number of data at each iteration
    eps = tolerance
    S_D = summary statistic of the data of point 4
    intercept = beta_0 obtained in point 4
    coefficients = beta_i, i=1,...,9 obtained in point 4
    '''
    theta = np.zeros((N_max + 1, 4))
    theta_0 = [0.07, 1.15, 0.05, 0.33]
    #theta_0 = [5., 5.0, 5.0, 5.0]
    theta[0,:] = theta_0
    
    prior_K_e = st.lognorm(s=0.6, scale=np.exp(-2.7))
    prior_K_a = st.lognorm(s=0.4, scale=np.exp(0.14))
    prior_Cl = st.lognorm(s=0.8, scale=np.exp(-3))
    prior_sigma = st.lognorm(s=0.3, scale=np.exp(-1.1))
    
    acceptance_count = 0
    rejection_count = 0
    entered = 0
    i = 0
    
    while i < N_max:
        # sample candidate parameters from a proposal transition density q(theta_i, )
        # anisotropic approach
        
        q_i_list = []
        theta_star_list = []
        s_list = [0.6, 0.4, 0.8, 0.3]
        for j in range(4):
            q_i = st.lognorm(s=s_list[j], scale=theta[i,j]) #lognormal 4 dimensional with different scale and same variance s
            q_i_list.append(q_i)
            theta_star=q_i.rvs() #sample from the lognormal
            theta_star_list.append(theta_star)
        
        # generate data from the underlying model given theta_star
        K_e = theta_star_list[0]
        K_a = theta_star_list[1]
        Cl = theta_star_list[2]
        sigma = theta_star_list[3]
        
        D_star = simulate_Xt_times(K_e, K_a, Cl, sigma) #generate data from underlying model given theta star
        
        if new_discrepancy_metric(D_star, S_D, intercept, coefficients, theta_0) < eps:
            
            prior_K_e_star = st.lognorm(s=s_list[0], scale=theta_star_list[0])
            prior_K_a_star = st.lognorm(s=s_list[1], scale=theta_star_list[1])
            prior_Cl_star = st.lognorm(s=s_list[2], scale=theta_star_list[2])
            prior_sigma_star = st.lognorm(s=s_list[3], scale=theta_star_list[3])
            
            q_star_pdf = jointly_prior_sample(prior_K_e_star, prior_K_a_star, prior_Cl_star, prior_sigma_star, theta[i,:])
            q_i_pdf = jointly_prior_sample(q_i_list[0], q_i_list[1], q_i_list[2], q_i_list[3], theta_star_list)
            pi_star = jointly_prior_sample(prior_K_e, prior_K_a, prior_Cl, prior_sigma, theta_star_list)
            pi_i = jointly_prior_sample(prior_K_e, prior_K_a, prior_Cl, prior_sigma, theta[i,:])
        
            comp = pi_star * q_star_pdf/(pi_i * q_i_pdf)
            comp = min(1,comp)
            
            #comp=0.85
            entered += 1
            
            if st.uniform.rvs() < comp:
                theta[i+1,:] = theta_star_list
                acceptance_count += 1 
            else:
                theta[i+1,:] = theta[i,:]
                rejection_count += 1
        else:
            theta[i+1,:] = theta[i,:]
            rejection_count += 1
            
        i += 1

    print('acceptance rate:', acceptance_count/(acceptance_count+rejection_count))
    return theta, acceptance_count/(acceptance_count+rejection_count)


def simulate_Xt_times_AV(K_e, K_a, Cl, sigma):
    '''
    Simulation of the ANTITHETIC pharmacokinetics model with Euler-Maruyama.
    Return only the values values of the stochastic process X_t for the times we are interested in.
    '''
    dt = 0.05  # Time step
    T = 12  # Total time
    n = int(T / dt)+1  # Number of time steps
    D = 4

    t = np.arange(0, 12.05, 0.05)  # Vector of times

    X_t = np.zeros(n) # Recall that X_0 = 0
    for i in range(n - 1):
        X_t[i + 1] = X_t[i] + dt*( (D*K_a*K_e)/Cl * np.exp(-K_a*t[i+1]) - K_e*X_t[i] ) - sigma*np.sqrt(dt)*np.random.randn()
    
    times = [0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12]
    indices = np.where(np.isin(t, times))
    Data = X_t[indices]
    return Data


def simulate_Xt_times_AV_full(K_e, K_a, Cl, sigma):
    '''
    Simulation of the ANTITHETIC pharmacokinetics model with Euler-Maruyama.
    Return all the times and all the values of the stochastic process X_t for each time evaluated.
    '''
    dt = 0.05  # Time step
    T = 12  # Total time
    n = int(T / dt)+1  # Number of time steps
    D = 4

    t = np.arange(0, 12.05, 0.05)  # Vector of times

    X_t = np.zeros(n) # Recall that X_0 = 0
    for i in range(n - 1):
        X_t[i + 1] = X_t[i] + dt*( (D*K_a*K_e)/Cl * np.exp(-K_a*t[i+1]) - K_e*X_t[i] ) - sigma*np.sqrt(dt)*np.random.randn()

    return t, X_t