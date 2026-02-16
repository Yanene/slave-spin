#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import os 
from matplotlib.ticker import FormatStrFormatter

def estimate_gauge(density):
    """Calculates the gauge term for the generic spin matrices """
    return( 1 / np.sqrt(density * (1 - density))) - 1

def fermi_dist(energy, beta):
    energy = np.asarray(energy).real
    beta = float(beta)
    exponent = beta * energy 

    if np.isscalar(exponent):  # Scalar case
        return 1. / (np.exp(exponent) + 1.) if exponent < 10. else 0
    else:  # Array case
        vector = np.zeros_like(exponent)
        small_indices = exponent < 10.
        vector[small_indices] = 1. / (np.exp(exponent[small_indices]) + 1.)
        return vector

def density_of_states(epsilon):
    "The DOS of Bethe lattice"
    return( 1 / (2 * np.pi * t**2)) * np.sqrt(np.clip(4 * t**2 - epsilon**2, 0, None))  # Clip to avoid sqrt of negative values

def calculate_h( mu, lambd, Z):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for x in xi:
            result[i]= density_of_states(x) * x * fermi_dist(Z*x-mu-lambd, beta)
            i=i+1
        return result
    x=np.linspace(-1,1,cut_sim)
    
    result= simpson(y1(x), x=x)
    return np.sqrt(Z) * result

# Function to calculate the average particle (fermions) number <n^f>
def fermionic_occupation(Z, mu, lambd):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for x in xi:
            result[i]= density_of_states(x) * fermi_dist(Z*x-mu-lambd, beta)
            i=i+1
        return result
    x=np.linspace(-1,1,cut_sim)
    
    result= simpson(y1(x), x=x)
    return result

def btest(state, index):
    """A bit test that evaluates if 'state' in binany has a one in the
       'index' location. returns one if true"""
    return (state >> index) & 1


def S_z( index):
    """Generates the spin_z operator """
    mat = np.zeros((4, 4))

    for i in range(4):
        ispin = btest(i, index)
        if ispin == 1:
            mat[i, i] = 1
        else:
            mat[i, i] = -1
    return 1/2.*mat


def O( index, gauge):
    """Generates the generic spin operator in z basis """
    mat = np.zeros((4, 4), dtype=np.complex128)

    flipper = 2**index
    for i in range(4):
        ispin = btest(i, index)
        if ispin ==1:
            mat[i ^ flipper, i] = 1
            
        else:
            mat[i ^ flipper, i] = gauge
           
    return mat

def O_dagger(index, gauge):
    """Generates the dagger (adjoint) of the generic spin operator in z basis """
    return O(index, gauge).conj().T

def O_O_dagger(index, gauge):
    return np.dot(O( index,gauge) ,O_dagger( index,gauge))


def average_value(operator, eigenvectors,eigenvalues):
    ''' This function calculates the average of an operator'''
    ground_state = eigenvectors[:,np.argmin(eigenvalues)]# take the eigenvector relative to the lowest eigenvalue
    average = np.dot(np.conjugate(ground_state), np.dot(operator, ground_state))# Calculate the average using np.dot
    return np.real(average)


def H_slave(h_value, lamda, U, gauge):
    # Define spin operators and identity matrix
    spin_op_0, spin_op_1, identity_matrix = S_z(0), S_z(1), np.identity(4)
    # Construct the Hamiltonian matrix
    matrix = h_value * (O_dagger(0, gauge) + O(0, gauge)) + np.conjugate(h_value) * (O_dagger(1, gauge) + O(1, gauge))
    matrix += lamda * (spin_op_0 + spin_op_1 + identity_matrix)
    matrix += (U / 2) * (spin_op_0 + spin_op_1) ** 2
    return matrix

def diagonalizer(A):
    eigvalues, eigvectors = scipy.linalg.eigh(A)
    return eigvalues, eigvectors

def numerical_gradient(f, x, h=1e-5):
    df = (f(x + h) - f(x-x)) / (2*h)
    return df

def gradient_descent(func, start, learn_rate, n_iter=5000, tolerance=1e-8):
    vector = start 
    for i in range(n_iter):
        x = vector
        grad = numerical_gradient(func, x)
        if np.all(np.abs(grad) <= tolerance):
            break
        vector -= learn_rate * grad
    return vector

def find_lambda(n_f, old_lambda, U, h_value, gauge):
    def spread_func(lamda):
            H = H_slave(h_value, lamda, U, gauge)
            evals, eigenvectors = np.linalg.eigh(H)
            average_Sz = average_value(S_z(0), eigenvectors, evals)
            spin_occupation = np.real(average_Sz) + 0.5
            spread_func_value = (spin_occupation -n_f)**2
            return spread_func_value

    start = old_lambda
    learn_rate = 0.1
    result = gradient_descent(spread_func, start, learn_rate)
    return result

def dens(m_n, m_c, l_n, Z):
    ''' This function calculate the density dn/dmu'''
    n_c = fermionic_occupation(Z,  m_c, l_n)
    n_n = fermionic_occupation(Z,  m_n, l_n)

    if m_c == m_n:
        return 0
    density = (n_c - n_n) / (m_c - m_n)
    if density > 0:
        return max(0.1, density)
    else:
        return min(-0.1, density)


def find_mu(m_c, lamda, d,Z,targ_occupation):
    """Routine to find the chemical potential at each loop."""
    n_c = fermionic_occupation(Z,  m_c, lamda)

    return m_c if d==0 else m_c - (n_c - targ_occupation) / d

def quasiparticle_weight(average):
    """Calculates the quasiparticle weight"""
    return average ** 2


def Self_consistency_loop(U_values, Z_guess, mu_guess, lamda_guess, dens_guess, targ_occupation, tol, filename):
    h_guess = -0.2
    for i, U in enumerate(U_values):
        bool = 0
        iteration = 0
        
        while bool == 0:
            iteration += 1
            
            fermionic_occ = fermionic_occupation(Z_guess, mu_guess, lamda_guess)
            gauge = estimate_gauge(fermionic_occ)
            h_new = calculate_h( mu_guess, lamda_guess, Z_guess)
            lamda_new = find_lambda(fermionic_occ, lamda_guess, U, h_new, gauge)

            evals, evecs = diagonalizer(H_slave(h_new, lamda_new, U, gauge))
            average_n = average_value(O(0, gauge), evecs, evals)
            Z_n = quasiparticle_weight(average_n)
            average_Sz_n = average_value(S_z(0), evecs, evals)
            spin_occupation_n = np.real(average_Sz_n) + 0.5
            
            
            mu_new = find_mu(mu_guess, lamda_new,  dens_guess, Z_n,targ_occupation)
            dens_new = dens(mu_new, mu_guess, lamda_new,Z_n)
            
            diff_lamda = lamda_new - lamda_guess
            diff_mu = mu_new - mu_guess
            diff_h = h_new - h_guess
            diff_Z = Z_n - Z_guess
            
            
            if (
                abs(diff_lamda) < tol and abs(diff_h) < tol and abs(diff_Z) < tol
                and abs(diff_mu) < tol
                and abs(spin_occupation_n - fermionic_occ) < tol
                and abs(targ_occupation - fermionic_occ) < tol 
            ):
        
                row=[U, mu_new, Z_n,lamda_new,
                    h_new,
                    fermionic_occ,
                    spin_occupation_n
                ]
                save_table([row],filename)
                bool = 1  # Exit the while loop
                print('####Succcessss####; U', U)
            else:
                print('U', U)
                print('diff_Z', diff_Z)
                print('diff_mu', diff_mu)
                print('diff_l', diff_lamda)
                print('diff_h', diff_h)
                print('targ_occ', abs(targ_occupation - fermionic_occ))
                print('dif_s_f_occ', abs(spin_occupation_n - fermionic_occ))
                alpha = 0.1
                h_guess = h_new * alpha + (1 - alpha) * h_guess
                lamda_guess = lamda_new * alpha + (1 - alpha) * lamda_guess
                mu_guess = mu_new * alpha + (1 - alpha) * mu_guess
                dens_guess = dens_new * alpha + (1 - alpha) * dens_guess
                Z_guess = Z_n * alpha + (1 - alpha) * Z_guess
                



def save_table(table, filename):
    header="# U mu  Z lamda h n_f n_s  \n"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:   
            file.write(header)
    with open (filename,'a') as file:        
        for row in table:
            file.write(" ".join(map(str, row)) + "\n")

def open_file(filename):
    if os.name == 'nt':
        os.startfile(filename)
    elif os.uname().sysname == 'Darwin':  
        os.system(f'open {filename}')
    else:
        os.system(f'xdg-open {filename}')

# Example run:
# Define constants
t = 0.5  # Hopping amplitude
D = 2.0 * t  # Half-bandwidth, energy unit
U_values = np.arange(0,3.6,0.1)  # values of the onsite coulomb repulsion
tol = 1e-5  # the Tolerance
beta = 1000
cut_sim=2000
targ_occupation = 0.5# half fillling 
# Initial seed
mu_guess = 0
Z_guess=1
lamda_guess = 0
dens_guess = 1
file_path = 'result_para_bethe_mudoc.dat'

Self_consistency_loop(U_values, Z_guess, mu_guess, lamda_guess, dens_guess, targ_occupation, tol, file_path)


