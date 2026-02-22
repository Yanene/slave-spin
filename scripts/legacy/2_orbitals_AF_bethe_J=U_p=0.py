import numpy as np
import os
from scipy.integrate import simpson


"""The eigenvalues of the fermionic Hamiltonian"""

def Lamda_k_plus(epsilon_k,lamda_tild_up,lamda_tild_down,Z_up,Z_down):
        return (-(lamda_tild_up+lamda_tild_down)+np.sqrt((lamda_tild_up-lamda_tild_down)**2+4*Z_up*Z_down*epsilon_k**2))/2


def Lamda_k_moins( epsilon_k,lamda_tild_up,lamda_tild_down,Z_up,Z_down):
        return (-(lamda_tild_up+lamda_tild_down)-np.sqrt((lamda_tild_up-lamda_tild_down)**2+4*Z_up*Z_down*epsilon_k**2))/2

"""The coefficient of the eigenvectors of the fermionic Hamiltonian correspond to Lamda_k_moins"""

def alpha_moins(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down):
        numerator = np.sqrt(Z_up * Z_down) * epsilon  * Lamda_k_moins( epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) 
        denominator = np.sqrt(Z_up * Z_down * epsilon ** 2 * Lamda_k_moins( epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) ** 2 + (Z_up * Z_down * epsilon ** 2 - lamda_tild_down * Lamda_k_moins( epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) - lamda_tild_up * lamda_tild_down) ** 2)
        result = numerator / denominator
        return result
    
def beta_moins(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down):
        numerator = (Z_up * Z_down * epsilon ** 2 - lamda_tild_down * Lamda_k_moins(epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) - lamda_tild_up * lamda_tild_down) 
        denominator = np.sqrt(Z_up * Z_down * epsilon ** 2 * Lamda_k_moins( epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) ** 2 + (Z_up * Z_down * epsilon ** 2 - lamda_tild_down * Lamda_k_moins( epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) - lamda_tild_up * lamda_tild_down) ** 2)
        result = numerator / denominator
        return result
    
    
"""The coefficient of the eigenvectors correspond to Lamda_k_plus"""

def alpha_plus(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down):
        numerator = np.sqrt(Z_up * Z_down) * epsilon  * Lamda_k_plus( epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) 
        denominator = np.sqrt(Z_up * Z_down * epsilon ** 2 * Lamda_k_plus( epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) ** 2 + (Z_up * Z_down * epsilon ** 2 - lamda_tild_down * Lamda_k_plus( epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) - lamda_tild_up * lamda_tild_down) ** 2)
        result = numerator / denominator
        return result
    
def beta_plus(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down):
        numerator = (Z_up * Z_down * epsilon ** 2 - lamda_tild_down * Lamda_k_plus(epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) - lamda_tild_up * lamda_tild_down) 
        denominator=np.sqrt(Z_up * Z_down * epsilon ** 2 * Lamda_k_plus( epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) ** 2 + (Z_up * Z_down * epsilon ** 2 - lamda_tild_down * Lamda_k_plus( epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down) - lamda_tild_up * lamda_tild_down) ** 2)
        result = numerator / denominator
        return result

def density_of_states(epsilon):
    "The DOS of Bethe lattice"
    return (1 / (2 * np.pi * t**2)) * np.sqrt(np.clip(4 * t**2 - epsilon**2, 0, None))  

def fermi_dist(energy,mu, beta):
    "Fermi_Dirac distribution function"
    beta = float(beta)
    mu = float(mu)

    exponent = beta * (energy-mu)
    if exponent < 100. :
        vector=1. / (np.exp(exponent) + 1.) 
    else:
        vector=0
    
    return vector   

"""The calculus of the fermionic occupation number for each site A and B"""
   
def fermionic_up(lamda_tild_up,lamda_tild_down, Z_up,Z_down,mu):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for x in xi:
            
            result[i]= density_of_states(x) * (np.linalg.norm(beta_moins(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)/(alpha_plus(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)*beta_moins(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)-alpha_moins(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)*beta_plus(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)))**2)*fermi_dist(Lamda_k_plus(x,lamda_tild_up,lamda_tild_down,Z_up,Z_down),mu, beta)
            i=i+1
        return result
    x=np.linspace(-1,1,cut_sim)
    
    result1= simpson(y1(x), x=x)
    def y2( xi): 
        result=np.zeros(len(xi))
        i=0
        for x in xi:
            result[i]= density_of_states(x)*np.linalg.norm(beta_plus(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)/(alpha_plus(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)*beta_moins(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)-alpha_moins(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)*beta_plus(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)))**2*fermi_dist(Lamda_k_moins(x,lamda_tild_up,lamda_tild_down,Z_up,Z_down),mu, beta)
            i=i+1
        return result

    result2= simpson(y2( x), x=x)

    return  (result1+result2)  


def fermionic_down(lamda_tild_up,lamda_tild_down, Z_up,Z_down,mu):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for x in xi:
            result[i]= density_of_states(x) *(np.linalg.norm(alpha_moins(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)/(beta_plus(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)*
                                                                                                                                   alpha_moins(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)-alpha_plus(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)
                                                                                                                                   *beta_moins(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)))**2)*fermi_dist(Lamda_k_plus(x,lamda_tild_up,lamda_tild_down,Z_up,Z_down),mu, beta)
            i=i+1
        return result
    x=np.linspace(-1,1,cut_sim)
    
    result1= simpson(y1( x), x=x)
    def y2( xi): 
        result=np.zeros(len(xi))
        i=0
        for x in xi:
            result[i]= density_of_states(x) *(np.linalg.norm(alpha_plus(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)/(beta_plus(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)*alpha_moins(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)-alpha_plus(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)*beta_moins(Z_up, Z_down, x, lamda_tild_up, lamda_tild_down)))**2)*fermi_dist(Lamda_k_moins(x,lamda_tild_up,lamda_tild_down,Z_up,Z_down),mu, beta)
            i=i+1
        return result
   
    result2= simpson(y2( x), x=x)

    return (result2+result1)    

"""The calculus of the parameter h for each site up and down"""

def calculate_h_up( lamda_tild_up,lamda_tild_down, Z_up,Z_down,mu):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for epsilon in xi:
            result[i]= density_of_states(epsilon) *epsilon*((alpha_plus(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down)*beta_plus(Z_up, Z_down,epsilon, lamda_tild_up, lamda_tild_down))/((alpha_plus(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down)*beta_moins(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down)-alpha_moins(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down)*beta_plus(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down))**2))*(fermi_dist(Lamda_k_plus(epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down),mu, beta)-fermi_dist(Lamda_k_moins(epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down),mu, beta))
            i=i+1
        return result
    epsilon=np.linspace(-1,1,cut_sim)
    
    result1= simpson(y1( epsilon), x=epsilon)

    return np.sqrt(Z_down)*result1


def calculate_h_down(lamda_tild_up,lamda_tild_down, Z_up,Z_down,mu):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for epsilon in xi:
            result[i]= density_of_states(epsilon) *epsilon*((alpha_plus(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down)*beta_plus(Z_up, Z_down,epsilon, lamda_tild_up, lamda_tild_down))/((alpha_plus(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down)*beta_moins(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down)-alpha_moins(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down)*beta_plus(Z_up, Z_down, epsilon, lamda_tild_up, lamda_tild_down))**2))*(fermi_dist(Lamda_k_plus(epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down),mu, beta)-fermi_dist(Lamda_k_moins(epsilon,lamda_tild_up,lamda_tild_down,Z_up,Z_down),mu,beta))
            i=i+1
        return result
    epsilon=np.linspace(-1,1,cut_sim)
    
    result1= simpson(y1( epsilon), x=epsilon)

    return np.sqrt(Z_up)*result1


def estimate_gauge(density):
    """Calculates the gauge term for the generic spin matrices """
    return (1/np.sqrt(density*(1-density)+0.00000001)) - 1

def btest(state, index):
    """Test if the bit at position 'index' in 'state' is set."""
    #state >> index -> shifts bits right
    #& 1 -> keeps only the last bit
    return (state >> index) & 1 

def S_z(orbitals, index):
    """Generates the spin_z operator for spin-orbital site 'index'."""
    dim = 4 ** orbitals
    mat = np.zeros((dim, dim))
    for i in range(dim):
        spin = btest(i, index)
        mat[i, i] = 0.5 if spin == 1 else -0.5
    return mat

def O(orbitals, index, gauge):
    """
    Spin-flip operator with a gauge:
    - If |↑⟩ → |↓⟩, coefficient = 1
    - If |↓⟩ → |↑⟩, coefficient = gauge
    """
    dim = 4 ** orbitals
    mat = np.zeros((dim, dim), dtype=np.complex128)
    flipper = 2 ** index  # flips the bit at position `index`
    
    for i in range(dim):
        spin = btest(i, index)
        j = i ^ flipper  # flip the spin at index
        if spin == 1:
            mat[j, i] = 1.0       # ↑ → ↓
        else:
            mat[j, i] = gauge     # ↓ → ↑ (with gauge)
    return mat


def O_dagger(orbitals, index, gauge):
    """Hermitian conjugate of O."""
    return O(orbitals, index, gauge).conj().T

def O_O_dagger(orbitals, index, gauge):
    """O O† operator product."""
    return np.dot(O(orbitals, index, gauge), O_dagger(orbitals, index, gauge))

def average_value(operator, eigenvectors, eigenvalues):
    """Computes ⟨ψ₀|O|ψ₀⟩ for the ground state."""
    ground_state = eigenvectors[:, np.argmin(eigenvalues)]
    return np.real(np.dot(np.conjugate(ground_state), np.dot(operator, ground_state)))

def H_slave(h1_up,h1_down, h2_up,h2_down, lamda1_up,lamda1_down, lamda2_up,lamda2_down, U, gauge1_up,gauge1_down, gauge2_up,gauge2_down, J, orbitals):
    """
    h1, h2: Renormalized kinetic energy terms
    lambda1, lambda2: Lagrange multipliers
    U: Hubbard repulsion
    J: Hund's coupling
    gauge1, gauge2: off-diagonal gauge parameters
    """
    dim = 4 ** orbitals
    H = np.zeros((dim, dim), dtype=np.complex128)

    # Indices: 0 (↓1), 1 (↑1), 2 (↓2), 3 (↑2)
    # Lagrange λ terms
    H +=  (lamda1_down *S_z(orbitals, 0) + lamda1_up *S_z(orbitals, 1))
    H +=  (lamda2_down *S_z(orbitals, 2) + lamda2_up *S_z(orbitals, 3))

    # Hopping terms
    H += (h1_down*O_dagger(orbitals, 0, gauge1_down) + h1_up*O_dagger(orbitals, 1, gauge1_up))
    H +=  (np.conj(h1_down) *O(orbitals, 0, gauge1_down) + np.conj(h1_up) *O(orbitals, 1, gauge1_up))
    H +=  (h2_down *O_dagger(orbitals, 2, gauge2_down) + h2_up *O_dagger(orbitals, 3, gauge2_up))
    H += (np.conj(h2_down) *O(orbitals, 2, gauge2_down) + np.conj(h2_up) *O(orbitals, 3, gauge2_up))

    # Define spin-z operators
    Sz_1_up   = S_z(orbitals, 1)
    Sz_1_down = S_z(orbitals, 0)
    Sz_2_up   = S_z(orbitals, 3)
    Sz_2_down = S_z(orbitals, 2)

    # Interaction terms
    # On-site U: same orbital, opposite spin
    H += U * (Sz_1_up @ Sz_1_down + Sz_2_up @ Sz_2_down)

    # Interorbital terms
    U_p = U - 2*J
    #H += U_p * (Sz_1_up @ Sz_2_down + Sz_1_down @ Sz_2_up)

    U_pp = U - 3*J
    #H += U_pp * (Sz_1_up @ Sz_2_up + Sz_1_down @ Sz_2_down)

    return H

def diagonalizer(A):
    eigvalues,eigvectors = np.linalg.eigh(A)
    return eigvalues,eigvectors    

def eta(density):
    return (2*density - 1 )/(4*density*(1-density)+0.00000001)

def quasiparticle_weight(average):
    return average**2
"""def quasiparticle_weight(average):
    return average**2"""

def lamda_0(h,average,eta):
    return 4*h*average*eta

def lamda_tild(lamda,lamda_0):
    return lamda-lamda_0


def numerical_gradient(f, x, h=1e-5):
    """
    Compute numerical gradient of f at vector x.
    f: function that takes a vector and returns a scalar
    x: numpy array (shape: [n])
    """
    grad = np.zeros_like(x, dtype=float)
    fx = f(x)  # base function value
    for i in range(len(x)):
        x_step = np.copy(x)
        x_step[i] += h
        grad[i] = (f(x_step) - fx) / h
    return grad


def gradient_descent(func, start, learn_rate, n_iter=5000, tolerance=1e-8):
    """
    Generic gradient descent for N-dimensional optimization
    """
    vector = np.array(start, dtype=float)
    for i in range(n_iter):
        grad = numerical_gradient(func, vector)
        func_value = func(vector)
        if np.abs(func_value) <= tolerance:
            break
        vector -= learn_rate * grad
    return vector


def find_lambdas(
    old_lambda1_up, old_lambda1_down, old_lambda2_up, old_lambda2_down,
    U, h1_up, h1_down, h2_up, h2_down,
    gauge1_up, gauge1_down, gauge2_up, gauge2_down,
    nm1_up, nm1_down, nm2_up, nm2_down,
    J, orbitals
):
    def spread_func(lambdas):
        lamda1_up, lamda1_down, lamda2_up, lamda2_down = lambdas

        H = H_slave(
            h1_up, h1_down, h2_up, h2_down,
            lamda1_up, lamda1_down, lamda2_up, lamda2_down,
            U, gauge1_up, gauge1_down, gauge2_up, gauge2_down,
            J, orbitals
        )
        evals, eigenvectors = np.linalg.eigh(H)

        # Site 1 occupations
        average_Sz_1up   = average_value(S_z(orbitals, 1), eigenvectors, evals)
        spin_occ_1up     = np.real(average_Sz_1up) + 0.5
        average_Sz_1down = average_value(S_z(orbitals, 0), eigenvectors, evals)
        spin_occ_1down   = np.real(average_Sz_1down) + 0.5

        # Site 2 occupations
        average_Sz_2up   = average_value(S_z(orbitals, 3), eigenvectors, evals)
        spin_occ_2up     = np.real(average_Sz_2up) + 0.5
        average_Sz_2down = average_value(S_z(orbitals, 2), eigenvectors, evals)
        spin_occ_2down   = np.real(average_Sz_2down) + 0.5

        # Spread (error function) across both sites
        spread_func_value = (
            (spin_occ_1up   - nm1_up)**2 +
            (spin_occ_1down - nm1_down)**2 +
            (spin_occ_2up   - nm2_up)**2 +
            (spin_occ_2down - nm2_down)**2
        )

        return spread_func_value

    # Initial guess
    start = np.array([old_lambda1_up, old_lambda1_down, old_lambda2_up, old_lambda2_down])
    learn_rate = 1
    result = gradient_descent(spread_func, start, learn_rate)

    return result[0], result[1], result[2], result[3]


def dens(m_n, m_c, lamda_tild1_up_n,lamda_tild1_down_n, Z1_up,Z1_down, lamda_tild2_up_n,lamda_tild2_down_n, Z2_up,Z2_down):
    ''' This function calculate the density dn/dmu'''
    n_c = fermionic_up(lamda_tild1_up_n,lamda_tild1_down_n, Z1_up,Z1_down,m_c)+fermionic_down(lamda_tild1_up_n,lamda_tild1_down_n, Z1_up,Z1_down,m_c)+fermionic_up(lamda_tild2_up_n,lamda_tild2_down_n, Z2_up,Z2_down,m_c)+fermionic_down(lamda_tild2_up_n,lamda_tild2_down_n, Z2_up,Z2_down,m_c)
    n_n = fermionic_up(lamda_tild1_up_n,lamda_tild1_down_n, Z1_up,Z1_down,m_n)+fermionic_down(lamda_tild1_up_n,lamda_tild1_down_n, Z1_up,Z1_down,m_n)+fermionic_up(lamda_tild2_up_n,lamda_tild2_down_n, Z2_up,Z2_down,m_n)+fermionic_down(lamda_tild2_up_n,lamda_tild2_down_n, Z2_up,Z2_down,m_n)
    if m_c == m_n:
        return 0
    density = (n_c - n_n) / (m_c - m_n)
    if density > 0:
        return max(0.2, density)
    else:
        return min(-0.2, density)


def find_mu(m_c,  lamda_tild1_up_n,lamda_tild1_down_n, Z1_up,Z1_down, lamda_tild2_up_n,lamda_tild2_down_n, Z2_up,Z2_down,d):
    """Routine to find the chemical potential at each loop."""
    n_c = fermionic_up(lamda_tild1_up_n,lamda_tild1_down_n, Z1_up,Z1_down,m_c)+fermionic_down(lamda_tild1_up_n,lamda_tild1_down_n, Z1_up,Z1_down,m_c)+fermionic_up(lamda_tild2_up_n,lamda_tild2_down_n, Z2_up,Z2_down,m_c)+fermionic_down(lamda_tild2_up_n,lamda_tild2_down_n, Z2_up,Z2_down,m_c)
    return m_c if d==0 else m_c - (n_c - targ_occupation) / d


def Self_consistency_loop(U_values,targ_occupation,Z1_up_guess,Z1_down_guess,Z2_up_guess,Z2_down_guess,lamda1_up_guess,lamda1_down_guess,lamda2_up_guess,lamda2_down_guess,lamda1_0_up_guess,lamda1_0_down_guess,lamda2_0_up_guess,lamda2_0_down_guess,mu_guess,dens_guess,orbitals,tol,filename):
    h1_down_old=-0.2
    h1_up_old=-0.2
    h2_down_old=-0.2
    h2_up_old=-0.2
    spin_occupation1_down_o=0.5
    spin_occupation1_up_o=0.5
    spin_occupation2_down_o=0.5
    spin_occupation2_up_o=0.5

    for i,U in enumerate(U_values):
        J=0.0*U
        bool = 0
        iteration=0
        while bool == 0:
            iteration=iteration+1
            lamda_tild1_up_guess = lamda1_up_guess - lamda1_0_up_guess
            lamda_tild1_down_guess = lamda1_down_guess - lamda1_0_down_guess
            lamda_tild2_up_guess = lamda2_up_guess - lamda2_0_up_guess
            lamda_tild2_down_guess = lamda2_down_guess - lamda2_0_down_guess
            
            f1_up = fermionic_up(lamda_tild1_up_guess, lamda_tild1_down_guess, Z1_up_guess, Z1_down_guess,mu_guess)
            f1_down = fermionic_down(lamda_tild1_up_guess, lamda_tild1_down_guess, Z1_up_guess, Z1_down_guess,mu_guess)
            f2_up = fermionic_up(lamda_tild2_up_guess, lamda_tild2_down_guess, Z2_up_guess, Z2_down_guess,mu_guess)
            f2_down = fermionic_down(lamda_tild2_up_guess, lamda_tild2_down_guess, Z2_up_guess, Z2_down_guess,mu_guess)
            
            eta1_up = eta(f1_up)
            eta1_down = eta(f1_down)
            eta2_up = eta(f2_up)
            eta2_down = eta(f2_down)
            
            gauge1_up = estimate_gauge(f1_up)
            gauge1_down = estimate_gauge(f1_down)
            gauge2_up = estimate_gauge(f2_up)
            gauge2_down = estimate_gauge(f2_down)
            
            h1_new_up = calculate_h_up(lamda_tild1_up_guess, lamda_tild1_down_guess, Z1_up_guess, Z1_down_guess,mu_guess)
            h1_new_down = calculate_h_down(lamda_tild1_up_guess, lamda_tild1_down_guess, Z1_up_guess, Z1_down_guess,mu_guess)
            h2_new_up = calculate_h_up(lamda_tild2_up_guess, lamda_tild2_down_guess, Z2_up_guess, Z2_down_guess,mu_guess)
            h2_new_down = calculate_h_down(lamda_tild2_up_guess, lamda_tild2_down_guess, Z2_up_guess, Z2_down_guess,mu_guess)
            
            lamda1_up_new, lamda1_down_new,lamda2_up_new, lamda2_down_new = find_lambdas(lamda1_up_guess, lamda1_down_guess, lamda2_up_guess, lamda2_down_guess,
                U, h1_new_up, h1_new_down, h2_new_up, h2_new_down,
                gauge1_up, gauge1_down, gauge2_up, gauge2_down,
                f1_up, f1_down, f2_up, f2_down,
                J, orbitals
            )
            
            
            evas, evcs = diagonalizer(H_slave(
                h1_new_up, h1_new_down, h2_new_up, h2_new_down,
                lamda1_up_new, lamda1_down_new, lamda2_up_new, lamda2_down_new,
                U, gauge1_up, gauge1_down, gauge2_up, gauge2_down,
                J, orbitals
            ))



            average1_up_n = average_value(O(orbitals,1, gauge1_up), evcs, evas)
            average1_down_n = average_value(O(orbitals,0, gauge1_down), evcs, evas)
            average2_up_n = average_value(O(orbitals,3, gauge2_up), evcs, evas)
            average2_down_n = average_value(O(orbitals,2, gauge2_down), evcs, evas)
            
            Z1_up_n = quasiparticle_weight(average1_up_n)
            Z1_down_n = quasiparticle_weight(average1_down_n)
            Z2_up_n = quasiparticle_weight(average2_up_n)
            Z2_down_n = quasiparticle_weight(average2_down_n)
            
            average_Sz_1up   = average_value(S_z(orbitals, 1), evcs, evas)
            spin_occ_1up     = np.real(average_Sz_1up) + 0.5
            average_Sz_1down = average_value(S_z(orbitals, 0), evcs, evas)
            spin_occ_1down   = np.real(average_Sz_1down) + 0.5

            average_Sz_2up   = average_value(S_z(orbitals, 3), evcs, evas)
            spin_occ_2up     = np.real(average_Sz_2up) + 0.5
            average_Sz_2down = average_value(S_z(orbitals, 2), evcs, evas)
            spin_occ_2down   = np.real(average_Sz_2down) + 0.5
            
            m1 = abs(f1_up - f1_down)
            m2 = abs(f2_up - f2_down)

            lamda1_0_up = lamda_0(h1_new_up, average1_up_n, eta1_up)
            lamda1_0_down = lamda_0(h1_new_down, average1_down_n, eta1_down)
            lamda2_0_up = lamda_0(h2_new_up, average2_up_n, eta2_up)
            lamda2_0_down = lamda_0(h2_new_down, average2_down_n, eta2_down)
            lamda1_tild_up = lamda_tild(lamda1_up_new, lamda1_0_up)
            lamda1_tild_down = lamda_tild(lamda1_down_new, lamda1_0_down)
            lamda2_tild_up = lamda_tild(lamda2_up_new, lamda2_0_up)
            lamda2_tild_down = lamda_tild(lamda2_down_new, lamda2_0_down)
            
            mu_new = find_mu(mu_guess, lamda1_tild_up,lamda1_tild_down, Z1_up_n,Z1_down_n, lamda2_tild_up,lamda2_tild_down, Z2_up_n,Z2_down_n,dens_guess)
            dens_new = dens(mu_new, mu_guess, lamda1_tild_up,lamda1_tild_down, Z1_up_n,Z1_down_n, lamda2_tild_up,lamda2_tild_down, Z2_up_n,Z2_down_n)
            


            diff_lamda1_up = lamda1_up_new - lamda1_up_guess
            diff_lamda2_up = lamda2_up_new - lamda2_up_guess
            
            diff_lamda1_down = lamda1_down_new - lamda1_down_guess
            diff_lamda2_down = lamda2_down_new - lamda2_down_guess
            
            diff_h1_up = h1_up_old - h1_new_up
            diff_h2_up = h2_up_old - h2_new_up
            
            diff_h1_down = h1_down_old - h1_new_down
            diff_h2_down = h2_down_old - h2_new_down
            
            diff_lamda1_up_0 = lamda1_0_up - lamda1_0_up_guess
            diff_lamda2_up_0 = lamda2_0_up - lamda2_0_up_guess
            
            diff_lamda1_down_0 = lamda1_0_down - lamda1_0_down_guess
            diff_lamda2_down_0 = lamda2_0_down - lamda2_0_down_guess
            
            diff_Z1_up = Z1_up_n - Z1_up_guess
            diff_Z2_up = Z2_up_n - Z2_up_guess
            
            diff_Z1_down = Z1_down_n- Z1_down_guess
            diff_Z2_down = Z2_down_n- Z2_down_guess
            
            diff_mu = mu_new - mu_guess
            
            
            
            if (
            abs(diff_Z1_up)   < tol and abs(diff_Z2_up)   < tol and
            abs(diff_Z1_down) < tol and abs(diff_Z2_down) < tol and

            abs(diff_lamda1_up)   < tol and abs(diff_lamda2_up)   < tol and
            abs(diff_lamda1_down) < tol and abs(diff_lamda2_down) < tol and

            abs(diff_lamda1_up_0)   < tol and abs(diff_lamda2_up_0)   < tol and
            abs(diff_lamda1_down_0) < tol and abs(diff_lamda2_down_0) < tol and

            abs(diff_h1_up)   < tol and abs(diff_h2_up)   < tol and
            abs(diff_h1_down) < tol and abs(diff_h2_down) < tol and

            abs(diff_mu) < tol and

            abs(targ_occupation - (f1_up + f1_down + f2_up + f2_down)) < tol and

            abs(spin_occ_1up   - spin_occupation1_up_o)   < tol and
            abs(spin_occ_1down - spin_occupation1_down_o) < tol and
            abs(spin_occ_2up   - spin_occupation2_up_o)   < tol and
            abs(spin_occ_2down - spin_occupation2_down_o) < tol
            ): 
                
                row=[U,m1,m2,mu_new, Z1_up_n, Z1_down_n,Z2_up_n, Z2_down_n,lamda1_up_new,lamda2_up_new, lamda1_down_new,lamda2_down_new
                    ,h1_new_up,h1_new_down,h2_new_up,h2_new_down,
                    lamda1_0_up, lamda1_0_down,lamda2_0_up, lamda2_0_down, f1_up,
                    f1_down,f2_up,f2_down,
                    spin_occ_1up, spin_occ_1down,spin_occ_2up,spin_occ_2down
                ]
                save_table([row],filename)
                bool = 1
                # print("success" + ' U=' + str(U))
                # print("success" + ' mu=' + str(mu_new))
            else:
                # print('U=' + str(U)+' Iteration ='+str(iteration))
                # print('diff Z1_up   = ' + str(abs(diff_Z1_up))) 
                # print('diff Z2_up   = ' + str(abs(diff_Z2_up))) 
                # print('diff Z1_down = ' + str(abs(diff_Z1_down))) 
                # print('diff Z2_down = ' + str(abs(diff_Z2_down))) 
                # print('diff_h1_up   = ' + str(abs(diff_h1_up))) 
                # print('diff_h2_up   = ' + str(abs(diff_h2_up))) 
                # print('diff_h1_down = ' + str(abs(diff_h1_down))) 
                # print('diff_h2_down = ' + str(abs(diff_h2_down))) 
                # print('diff_lamda1_up   = ' + str(abs(diff_lamda1_up))) 
                # print('diff_lamda2_up   = ' + str(abs(diff_lamda2_up))) 
                # print('diff_lamda1_down = ' + str(abs(diff_lamda1_down))) 
                # print('diff_lamda2_down = ' + str(abs(diff_lamda2_down))) 
                # print('diff_lamda1_up_0   = ' + str(abs(diff_lamda1_up_0))) 
                # print('diff_lamda2_up_0   = ' + str(abs(diff_lamda2_up_0))) 
                # print('diff_lamda1_down_0 = ' + str(abs(diff_lamda1_down_0))) 
                # print('diff_lamda2_down_0 = ' + str(abs(diff_lamda2_down_0))) 
                # print('diff_fermionic_occ_target = ' +str(abs(targ_occupation - (f1_up + f1_down + f2_up + f2_down)))) 
                # print('diff_spin_occ_1up   = ' + str(abs(spin_occ_1up   - spin_occupation1_up_o))) 
                # print('diff_spin_occ_1down = ' + str(abs(spin_occ_1down - spin_occupation1_down_o))) 
                # print('diff_spin_occ_2up   = ' + str(abs(spin_occ_2up   - spin_occupation2_up_o))) 
                # print('diff_spin_occ_2down = ' + str(abs(spin_occ_2down - spin_occupation2_down_o))) 
                # print('diff_mu = ' + str(abs(diff_mu)))

                
                # Update variables for the next iteration with mixing:
                alpha=0.1
                # Update h fields
                h1_up_old   = h1_new_up   * alpha + (1 - alpha) * h1_up_old 
                h2_up_old   = h2_new_up   * alpha + (1 - alpha) * h2_up_old 
                h1_down_old = h1_new_down * alpha + (1 - alpha) * h1_down_old 
                h2_down_old = h2_new_down * alpha + (1 - alpha) * h2_down_old 
                # Update spin occupations 
                spin_occupation1_up_o   = spin_occ_1up   * alpha + (1 - alpha) * spin_occupation1_up_o 
                spin_occupation1_down_o = spin_occ_1down * alpha + (1 - alpha) * spin_occupation1_down_o 
                spin_occupation2_up_o   = spin_occ_2up   * alpha + (1 - alpha) * spin_occupation2_up_o 
                spin_occupation2_down_o = spin_occ_2down * alpha + (1 - alpha) * spin_occupation2_down_o 
                # Update lambda guesses 
                lamda1_up_guess   = lamda1_up_new   * alpha + (1 - alpha) * lamda1_up_guess 
                lamda2_up_guess   = lamda2_up_new   * alpha + (1 - alpha) * lamda2_up_guess 
                lamda1_down_guess = lamda1_down_new * alpha + (1 - alpha) * lamda1_down_guess 
                lamda2_down_guess = lamda2_down_new * alpha + (1 - alpha) * lamda2_down_guess 
                # Update lambda_0 guesses 
                lamda1_0_up_guess   = lamda1_0_up   * alpha + (1 - alpha) * lamda1_0_up_guess 
                lamda2_0_up_guess   = lamda2_0_up   * alpha + (1 - alpha) * lamda2_0_up_guess 
                lamda1_0_down_guess = lamda1_0_down * alpha + (1 - alpha) * lamda1_0_down_guess 
                lamda2_0_down_guess = lamda2_0_down * alpha + (1 - alpha) * lamda2_0_down_guess 
                # Update Z factors 
                Z1_up_guess   = Z1_up_n   * alpha + (1 - alpha) * Z1_up_guess 
                Z2_up_guess   = Z2_up_n   * alpha + (1 - alpha) * Z2_up_guess 
                Z1_down_guess = Z1_down_n * alpha + (1 - alpha) * Z1_down_guess 
                Z2_down_guess = Z2_down_n * alpha + (1 - alpha) * Z2_down_guess 
                # Update chemical potential and density 
                mu_guess   = mu_new   * alpha + (1 - alpha) * mu_guess 
                dens_guess = dens_new * alpha + (1 - alpha) * dens_guess





def save_table(table, filename):
    header="# U m1 m2 mu_new Z1_up_n Z1_down_n Z2_up_n Z2_down_n lamda1_up_new lamda2_up_new lamda1_down_new lamda2_down_new h1_new_up h1_new_down h2_new_up h2_new_down lamda1_0_up lamda1_0_down lamda2_0_up lamda2_0_down f1_up f1_down f2_up f2_down spin_occ_1up spin_occ_1down spin_occ_2up spin_occ_2down \n"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            file.write(header)

    # Append data
    with open(filename, 'a') as file:
        for row in table:
            file.write(" ".join(map(str, row)) + "\n")
def open_file(filename):
    if os.name == 'nt':
        os.startfile(filename)
    elif os.uname().sysname == 'Darwin':  
        os.system(f'open {filename}')
    else:
        os.system(f'xdg-open {filename}')

# Example usage:

tol = 1e-4  # the tolerance
t = 0.5  # Hopping amplitude
D = 2.0 * t  #half_downandwidth
beta=10000
orbitals=2
U_values = np.arange(0,1,0.1)
# Initial seed
cut_sim=2000 #cut off for simpson rule used in fermionic integrals the higher it is the more precise but also the slower it gets
lamda1_up_guess = -1.6
lamda1_down_guess = 1.6
lamda2_up_guess = -1.6
lamda2_down_guess = 1.6
lamda1_0_up_guess=-3.16
lamda1_0_down_guess=3.16
lamda2_0_up_guess=-3.16
lamda2_0_down_guess=3.16
targ_occupation=2
Z1_up_guess,Z1_down_guess=1,1
Z2_up_guess,Z2_down_guess=1,1
mu_guess=0
dens_guess=1


file_path = 'result_AF_bethe_2orbitals_J=U_p=0.dat' #'/Users/youssra/Desktop/result_upF_downethe_mu.dat'
Self_consistency_loop(U_values,targ_occupation,Z1_up_guess,Z1_down_guess,Z2_up_guess,Z2_down_guess,lamda1_up_guess,lamda1_down_guess,lamda2_up_guess,lamda2_down_guess,lamda1_0_up_guess,lamda1_0_down_guess,lamda2_0_up_guess,lamda2_0_down_guess,mu_guess,dens_guess,orbitals,tol,file_path)

