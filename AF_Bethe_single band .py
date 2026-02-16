import numpy as np
import scipy.linalg
import os
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


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

"""The eigenvalues of the fermionic Hamiltonian"""

def Lamda_k_plus(epsilon_k,lamda_tild_A,lamda_tild_B,Z_A,Z_B):
        return (-(lamda_tild_A+lamda_tild_B)+np.sqrt((lamda_tild_A-lamda_tild_B)**2+4*Z_A*Z_B*epsilon_k**2))/2


def Lamda_k_moins( epsilon_k,lamda_tild_A,lamda_tild_B,Z_A,Z_B):
        return (-(lamda_tild_A+lamda_tild_B)-np.sqrt((lamda_tild_A-lamda_tild_B)**2+4*Z_A*Z_B*epsilon_k**2))/2

"""The coefficient of the eigenvectors of the fermionic Hamiltonian correspond to Lamda_k_moins"""

def alpha_moins(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B):
        numerator = np.sqrt(Z_A * Z_B) * epsilon  * Lamda_k_moins( epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) 
        denominator = np.sqrt(Z_A * Z_B * epsilon ** 2 * Lamda_k_moins( epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) ** 2 + (Z_A * Z_B * epsilon ** 2 - lamda_tild_B * Lamda_k_moins( epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) - lamda_tild_A * lamda_tild_B) ** 2)
        result = numerator / denominator
        return result
    
def beta_moins(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B):
        numerator = (Z_A * Z_B * epsilon ** 2 - lamda_tild_B * Lamda_k_moins(epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) - lamda_tild_A * lamda_tild_B) 
        denominator = np.sqrt(Z_A * Z_B * epsilon ** 2 * Lamda_k_moins( epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) ** 2 + (Z_A * Z_B * epsilon ** 2 - lamda_tild_B * Lamda_k_moins( epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) - lamda_tild_A * lamda_tild_B) ** 2)
        result = numerator / denominator
        return result
    
    
"""The coefficient of the eigenvectors correspond to Lamda_k_plus"""

def alpha_plus(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B):
        numerator = np.sqrt(Z_A * Z_B) * epsilon  * Lamda_k_plus( epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) 
        denominator = np.sqrt(Z_A * Z_B * epsilon ** 2 * Lamda_k_plus( epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) ** 2 + (Z_A * Z_B * epsilon ** 2 - lamda_tild_B * Lamda_k_plus( epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) - lamda_tild_A * lamda_tild_B) ** 2)
        result = numerator / denominator
        return result
    
def beta_plus(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B):
        numerator = (Z_A * Z_B * epsilon ** 2 - lamda_tild_B * Lamda_k_plus(epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) - lamda_tild_A * lamda_tild_B) 
        denominator=np.sqrt(Z_A * Z_B * epsilon ** 2 * Lamda_k_plus( epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) ** 2 + (Z_A * Z_B * epsilon ** 2 - lamda_tild_B * Lamda_k_plus( epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B) - lamda_tild_A * lamda_tild_B) ** 2)
        result = numerator / denominator
        return result
    
"""The calculus of the fermionic occupation number for each site A and B"""
   
def fermionic_A(lamda_tild_A,lamda_tild_B, Z_A,Z_B,mu):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for x in xi:
            
            result[i]= density_of_states(x) * (np.linalg.norm(beta_moins(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)/(alpha_plus(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)*beta_moins(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)-alpha_moins(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)*beta_plus(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)))**2)*fermi_dist(Lamda_k_plus(x,lamda_tild_A,lamda_tild_B,Z_A,Z_B),mu, beta)
            i=i+1
        return result
    x=np.linspace(-1,1,cut_sim)
    
    result1= simpson(y1(x), x=x)
    def y2( xi): 
        result=np.zeros(len(xi))
        i=0
        for x in xi:
            result[i]= density_of_states(x)*np.linalg.norm(beta_plus(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)/(alpha_plus(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)*beta_moins(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)-alpha_moins(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)*beta_plus(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)))**2*fermi_dist(Lamda_k_moins(x,lamda_tild_A,lamda_tild_B,Z_A,Z_B),mu, beta)
            i=i+1
        return result

    result2= simpson(y2( x), x=x)

    return  (result1+result2)  


def fermionic_B(lamda_tild_A,lamda_tild_B, Z_A,Z_B,mu):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for x in xi:
            result[i]= density_of_states(x) *(np.linalg.norm(alpha_moins(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)/(beta_plus(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)*
                                                                                                                                   alpha_moins(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)-alpha_plus(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)
                                                                                                                                   *beta_moins(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)))**2)*fermi_dist(Lamda_k_plus(x,lamda_tild_A,lamda_tild_B,Z_A,Z_B),mu, beta)
            i=i+1
        return result
    x=np.linspace(-1,1,cut_sim)
    

    result1= simpson(y1( x), x=x)
    def y2( xi): 
        result=np.zeros(len(xi))
        i=0
        for x in xi:
            result[i]= density_of_states(x) *(np.linalg.norm(alpha_plus(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)/(beta_plus(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)*alpha_moins(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)-alpha_plus(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)*beta_moins(Z_A, Z_B, x, lamda_tild_A, lamda_tild_B)))**2)*fermi_dist(Lamda_k_moins(x,lamda_tild_A,lamda_tild_B,Z_A,Z_B),mu, beta)
            i=i+1
        return result
   
    result2= simpson(y2( x), x=x)

    return (result2+result1)    

"""The calculus of the parameter h for each site A and B"""

def calculate_h_A( lamda_tild_A,lamda_tild_B, Z_A,Z_B,mu):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for epsilon in xi:
            result[i]= density_of_states(epsilon) *epsilon*((alpha_plus(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B)*beta_plus(Z_A, Z_B,epsilon, lamda_tild_A, lamda_tild_B))/((alpha_plus(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B)*beta_moins(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B)-alpha_moins(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B)*beta_plus(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B))**2))*(fermi_dist(Lamda_k_plus(epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B),mu, beta)-fermi_dist(Lamda_k_moins(epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B),mu, beta))
            i=i+1
        return result
    epsilon=np.linspace(-1,1,cut_sim)
    
    result1= simpson(y1( epsilon), x=epsilon)


    return np.sqrt(Z_B)*result1


def calculate_h_B(lamda_tild_A,lamda_tild_B, Z_A,Z_B,mu):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for epsilon in xi:
            result[i]= density_of_states(epsilon) *epsilon*((alpha_plus(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B)*beta_plus(Z_A, Z_B,epsilon, lamda_tild_A, lamda_tild_B))/((alpha_plus(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B)*beta_moins(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B)-alpha_moins(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B)*beta_plus(Z_A, Z_B, epsilon, lamda_tild_A, lamda_tild_B))**2))*(fermi_dist(Lamda_k_plus(epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B),mu, beta)-fermi_dist(Lamda_k_moins(epsilon,lamda_tild_A,lamda_tild_B,Z_A,Z_B),mu,beta))
            i=i+1
        return result
    epsilon=np.linspace(-1,1,cut_sim)
    
    result1= simpson(y1( epsilon), x=epsilon)


    return np.sqrt(Z_A)*result1


def estimate_gauge(density):
    """Calculates the gauge term for the generic spin matrices """
    return (1/np.sqrt(density*(1-density)+0.00000001)) - 1

def btest(state, index):
    """A bit test that evaluates if 'state' in binany has a one in the
       'index' location. returns one if true"""
    return (state >> index) & 1


def S_z( index):
    """Generates the spin_z operator """
    mat = np.zeros((4, 4))# Initialize a 4x4 matrix for a two-spin Hilbert space
    
    for i in range(4): # Loop over all basis states |i>
        ispin = btest(i, index) # Determine the spin value (0 or 1) at position 'index'
        if ispin == 1:
            mat[i, i] = 1 # Assign +1 for spin up, -1 for spin down
        else:
            mat[i, i] = -1
    return 1/2.*mat


def O( index, gauge):
    """Generates the generic spin operator in z basis """
    mat = np.zeros((4, 4), dtype=np.complex128)# Initialize a complex 4x4 matrix

    flipper = 2**index # to flip the spin at position 'index'
    for i in range(4):# loop over all basis states |i>
        ispin = btest(i, index)# check the spin at position 'index'
        if ispin ==1: # flip the spin using bitwise XOR
            mat[i ^ flipper, i] = 1 # Spin down to spin up transition

        else:
            mat[i ^ flipper, i] = gauge # Spin up to spin down transition with gauge factor

    return mat


def O_dagger(index, gauge):
    """Generates the dagger (adjoint) of the generic spin operator in z basis """
    return O(index, gauge).conj().T # Complex conjugate transpose of O


def average_value(operator, eigenvectors,eigenvalues):
    ''' This function calculates the average of an operator'''
    ground_state = eigenvectors[:,np.argmin(eigenvalues)]# take the eigenvector relative to the lowest eigenvalue
    average = np.dot(np.conjugate(ground_state), np.dot(operator, ground_state))# Calculate the average using np.dot
    return np.real(average)

def H_slave(h_value_A,h_value_B, lamda_A,lamda_B, U, gauge_A,gauge_B):
    '''The spin Hamiltonian '''
    spin_op_A, spin_op_B = S_z(0), S_z(1)
    # Construct the Hamiltonian matrix
    matrix = h_value_A * O_dagger(0, gauge_A) + np.conj(h_value_A)*O(0, gauge_A) + h_value_B * O_dagger(1, gauge_B) + np.conj(h_value_B)*O(1, gauge_B) 
    matrix += lamda_A * spin_op_A+ lamda_B*spin_op_B
    matrix += U  * (spin_op_A *spin_op_B) 
    return matrix

def diagonalizer(A):
    eigvalues,eigvectors = np.linalg.eigh(A)
    return eigvalues,eigvectors    

def eta(density):
    return (2*density - 1 )/(4*density*(1-density)+0.00000001)

def quasiparticle_weight(average_A,average_B):
    return average_A*average_B #average**2
"""def quasiparticle_weight(average):
    return average**2"""

def lamda_0(h,average,eta):
    return 4*h*average*eta

def lamda_tild(lamda,lamda_0):
    return lamda-lamda_0


def numerical_gradient(f, x, y, h=1e-5):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    
    return np.array([df_dx, df_dy])

def gradient_descent(func, start, learn_rate, n_iter=5000, tolerance=1e-8):
    vector = np.array(start, dtype=float) # Convert starting point to a float numpy array for calculations
    for i in range(n_iter):
        x, y = vector
        grad = numerical_gradient(func, x, y)# Compute the numerical gradient at the current point
        func_value=func(vector[0],vector[1])# Evaluate the function value at the current point
        if np.all(np.abs(func_value) <= tolerance):# Check convergence: if function value is small enough, stop
            break
        vector -= learn_rate * grad # Update the current point by moving against the gradient
    return vector

def find_lambda(old_lambda_A, old_lambda_B, U, h_value_A, h_value_B, gauge_A, gauge_B,nm_up,nm_down):
    def spread_func(lamda_A,lamda_B):
            H = H_slave(h_value_A, h_value_B, lamda_A, lamda_B, U, gauge_A, gauge_B)
            evals, eigenvectors = np.linalg.eigh(H)
            average_Sz_up = average_value(S_z(0), eigenvectors, evals)
            spin_occupation_up = np.real(average_Sz_up) + 0.5
            average_Sz_down = average_value(S_z(1), eigenvectors, evals)
            spin_occupation_down = np.real(average_Sz_down) + 0.5

            spread_func_value = (spin_occupation_up -nm_up)**2+( spin_occupation_down - nm_down) ** 2 
            
            return spread_func_value

    start = np.array([old_lambda_A, old_lambda_B])  
    learn_rate = 1
    result = gradient_descent(spread_func, start, learn_rate)
    return result[0], result[1]
##### Smart estimator method to calculate the chemical potential ######
def dens(m_n, m_c, lamda_tild_A_n,lamda_tild_B_n, Z_A,Z_B):
    ''' This function calculate the density dn/dmu'''
    n_c = fermionic_A(lamda_tild_A_n,lamda_tild_B_n, Z_A,Z_B,m_c)+fermionic_B(lamda_tild_A_n,lamda_tild_B_n, Z_A,Z_B,m_c)
    n_n = fermionic_A(lamda_tild_A_n,lamda_tild_B_n, Z_A,Z_B,m_n)+fermionic_B(lamda_tild_A_n,lamda_tild_B_n, Z_A,Z_B,m_n)
    if m_c == m_n:
        return 0
    density = (n_c - n_n) / (m_c - m_n)
    if density > 0:
        return max(0.2, density)
    else:
        return min(-0.2, density)


def find_mu(m_c, lamda_tild_A_n,lamda_tild_B_n, Z_A,Z_B,d):
    """Routine to find the chemical potential at each loop."""
    n_c = fermionic_A(lamda_tild_A_n,lamda_tild_B_n, Z_A,Z_B,m_c)+fermionic_B(lamda_tild_A_n,lamda_tild_B_n, Z_A,Z_B,m_c)
    return m_c if d==0 else m_c - (n_c - targ_occupation) / d

######## Main slave spin loop ########

def Self_consistency_loop(U_values,targ_occupation,Z_A_guess,Z_B_guess,lamda_A_guess,lamda_B_guess,lamda_0_A_guess,lamda_0_B_guess,mu_guess,dens_guess,tol,filename):
    # Initial guesses
    h_B_old=-0.2
    h_A_old=-0.2
    spin_occupation_B_o=0.5
    spin_occupation_A_o=0.5
    for i,U in enumerate(U_values):
        bool = 0
        iteration=0
        while bool == 0:
            iteration=iteration+1

            lamda_tild_A_guess = lamda_A_guess - lamda_0_A_guess
            lamda_tild_B_guess = lamda_B_guess - lamda_0_B_guess

            f_a = fermionic_A(lamda_tild_A_guess, lamda_tild_B_guess, Z_A_guess, Z_B_guess,mu_guess)
            f_b = fermionic_B(lamda_tild_A_guess, lamda_tild_B_guess, Z_A_guess, Z_B_guess,mu_guess)

            eta_A = eta(f_a)
            eta_B = eta(f_b)

            gauge_A = estimate_gauge(f_a)
            gauge_B = estimate_gauge(f_b)

            h_new_A = calculate_h_A(lamda_tild_A_guess, lamda_tild_B_guess, Z_A_guess, Z_B_guess,mu_guess)
            h_new_B = calculate_h_B(lamda_tild_A_guess, lamda_tild_B_guess, Z_A_guess, Z_B_guess,mu_guess)

            lamda_A_new, lamda_B_new = find_lambda(lamda_A_guess, lamda_B_guess, U, h_new_A, h_new_B, gauge_A, gauge_B, f_a, f_b)

            evas, evcs = diagonalizer(H_slave(h_new_A, h_new_B, lamda_A_new, lamda_B_new, U, gauge_A, gauge_B))

            average_A_n = average_value(O(0, gauge_A), evcs, evas)
            average_B_n = average_value(O(1, gauge_B), evcs, evas)

            Z_A_n = quasiparticle_weight(average_A_n,average_B_n)
            Z_B_n = quasiparticle_weight(average_A_n,average_B_n)

            average_Sz_A_n = average_value(S_z(0), evcs, evas)
            average_Sz_B_n = average_value(S_z(1), evcs, evas)

            spin_occupation_A_n = np.real(average_Sz_A_n) + 0.5
            spin_occupation_B_n = np.real(average_Sz_B_n) + 0.5

            m = abs(f_a - f_b)

            lamda_0_A = lamda_0(h_new_A, average_A_n, eta_A)
            lamda_0_B = lamda_0(h_new_B, average_B_n, eta_B)

            lamda_tild_A = lamda_tild(lamda_A_new, lamda_0_A)
            lamda_tild_B = lamda_tild(lamda_B_new, lamda_0_B)

            mu_new = find_mu(mu_guess, lamda_tild_A,lamda_tild_B, Z_A_n,Z_B_n,dens_guess)
            dens_new = dens(mu_new, mu_guess, lamda_tild_A,lamda_tild_B, Z_A_n,Z_B_n)
            
            diff_lamda_A = lamda_A_new - lamda_A_guess
            diff_lamda_B = lamda_B_new - lamda_B_guess
            diff_h_A = h_A_old - h_new_A
            diff_h_B = h_B_old - h_new_B
            diff_lamda_A_0 = lamda_0_A - lamda_0_A_guess
            diff_lamda_B_0 = lamda_0_B - lamda_0_B_guess
            diff_Z_A = Z_A_n - Z_A_guess
            diff_Z_B = Z_B_n- Z_B_guess
            diff_mu = mu_new - mu_guess

            if (abs(diff_Z_A) < tol and abs(diff_Z_B) < tol and abs(diff_lamda_A) < tol and abs(diff_lamda_B) < tol and abs(diff_lamda_A_0) < tol and abs(diff_lamda_B_0) < tol 
                and abs(diff_h_A) < tol and abs(diff_h_B) < tol and abs(diff_mu)<tol
                and abs(targ_occupation - (f_a + f_b)) < tol and abs(spin_occupation_A_n - spin_occupation_A_o) < tol and abs(spin_occupation_B_n - spin_occupation_B_o) < tol):
                #m_values.append(m)
                row=[U,m,mu_new, Z_A_n, Z_B_n,lamda_A_new, lamda_B_new
                    ,h_new_A,h_new_B,
                    lamda_0_A, lamda_0_B, f_a,
                    f_b,
                    spin_occupation_A_n, spin_occupation_B_n
                ]
                save_table([row],filename)
                bool = 1
                print("success" + ' U=' + str(U))
                print("success" + ' mu=' + str(mu_new))
            else:
                print('U=' + str(U)+' Iteration ='+str(iteration))
                print('diff Za=' + str(abs(diff_Z_A)))
                print('diff Zb=' + str(abs(diff_Z_B)))
                print('diff_h_A=' + str(abs(diff_h_A)))
                print('diff_h_B=' + str(abs(diff_h_B)))
                print('diff_lamda_A=' + str(abs(diff_lamda_A)))
                print('diff_lamda_B=' + str(abs(diff_lamda_B)))
                print('diff_lamda_A_0=' + str(abs(diff_lamda_A_0)))
                print('diff_lamda_B_0=' + str(abs(diff_lamda_B_0)))
                print('diff_fermionic_occ_target=' + str(abs(targ_occupation - (f_a + f_b)) ))
                print('diff_spin_occ_A=' + str(abs(spin_occupation_A_n - spin_occupation_A_o)))
                print('diff_spin_occ_B=' + str(abs(spin_occupation_B_n - spin_occupation_B_o)))
                print('diff_mu=' + str(abs(diff_mu)))
                
                # Update variables for the next iteration with mixing:
                alpha=0.1
                h_A_old = h_new_A*alpha+(1-alpha)*h_A_old
                h_B_old= h_new_B*alpha+(1-alpha)*h_B_old
                spin_occupation_B_o=spin_occupation_B_n*alpha+(1-alpha)*spin_occupation_B_o
                spin_occupation_A_o=spin_occupation_A_n*alpha+(1-alpha)*spin_occupation_A_o
                lamda_A_guess = lamda_A_new*alpha+(1-alpha)*lamda_A_guess
                lamda_B_guess = lamda_B_new*alpha+(1-alpha)*lamda_B_guess
                lamda_0_A_guess = lamda_0_A*alpha+(1-alpha)*lamda_0_A_guess
                lamda_0_B_guess = lamda_0_B*alpha+(1-alpha)*lamda_0_B_guess
                Z_A_guess = Z_A_n*alpha+(1-alpha)*Z_A_guess
                Z_B_guess = Z_B_n*alpha+(1-alpha)*Z_B_guess
                mu_guess=mu_new*alpha+(1-alpha)*mu_guess
                dens_guess=dens_new*alpha+(1-alpha)*dens_guess
            




def save_table(table, filename):
    header = "# U m mu  Z_A Z_B lamda_A_new lamda_B_new h_new_A h_new_B lamda_0_A  lamda_0_B  n_f_A n_f_B n_s_A n_s_B  \n"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            file.write(header)
    with open(filename,'a')as file:
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
D = 2.0 * t  #half_bandwidth
beta=10000
m_values=[]
U_values = np.arange(0,4,0.1)
# Initial seed
cut_sim=2000 #cut off for simpson rule used in fermionic integrals the higher it is the more precise but also the slower it gets
lamda_A_guess = -1.6
lamda_B_guess = 1.6
lamda_0_A_guess=-3.16
lamda_0_B_guess=3.16
targ_occupation=1
Z_A_guess,Z_B_guess=1,1
mu_guess=0
dens_guess=1

file_path = 'result_AF_bethe_mu1.dat' 
Self_consistency_loop(U_values,targ_occupation,Z_A_guess,Z_B_guess,lamda_A_guess,lamda_B_guess,lamda_0_A_guess,lamda_0_B_guess,mu_guess,dens_guess,tol,file_path)

