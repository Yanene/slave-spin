import os
import numpy as np
import scipy.linalg
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def fermi(energy, beta):
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

def fermionic_occupation1(Z_1, mu, lambd_1):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for epsilon in xi:
            result[i]= density_of_states(epsilon) * fermi(Z_1*epsilon-mu-lambd_1, beta)
            i=i+1
        return result
    epsilon=np.linspace(-1,1,cut_sim)
    
    result1= simpson(y1( epsilon), x=epsilon)

    return result1


def fermionic_occupation2(Z_2, mu, lambd_2):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for epsilon in xi:
            result[i]= density_of_states(epsilon) * fermi(Z_2*epsilon-mu-lambd_2, beta)
            i=i+1
        return result
    epsilon=np.linspace(-1,1,cut_sim)
    
    result1= simpson(y1( epsilon), x=epsilon)

    return result1


def calculate_h1( mu, lambd_1, Z_1):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for epsilon in xi:
            result[i]= density_of_states(epsilon) * epsilon * fermi(Z_1*epsilon-mu-lambd_1, beta)
            i=i+1
        return result
    epsilon=np.linspace(-1,1,cut_sim)
    
    result1= simpson(y1( epsilon), x=epsilon)

    return np.sqrt(Z_1)*result1


def calculate_h2( mu, lambd_2, Z_2):
    def y1(xi): 
        result=np.zeros(len(xi))
        i=0
        for epsilon in xi:
            result[i]= density_of_states(epsilon) * epsilon * fermi(Z_2*epsilon-mu-lambd_2, beta)
            i=i+1
        return result
    epsilon=np.linspace(-1,1,cut_sim)
    
    result1= simpson(y1( epsilon), x=epsilon)

    return np.sqrt(Z_2)*result1

def estimate_gauge(density):
    """Calculates the gauge term for the generic spin matrices """
    return( 1 / np.sqrt(density * (1 - density)+0.00000001)) - 1


def btest(state, index):
    """Test if the bit at position 'index' in 'state' is set"""
    return (state >> index) & 1

def S_z(orbitals, index):
    """Generates the spin_z operator for spin-orbital site 'index'"""
    dim = 4 ** orbitals
    mat = np.zeros((dim, dim))
    for i in range(dim):
        spin = btest(i, index)
        mat[i, i] = 0.5 if spin == 1 else -0.5
    return mat

def O(orbitals, index, gauge):
    """
    Spin-flip operator with a gauge:
    - If |￪⟩ → |￬⟩, coefficient = 1
    - If |￬⟩ → |￪⟩, coefficient = gauge
    """
    dim = 4 ** orbitals
    mat = np.zeros((dim, dim), dtype=np.complex128)
    flipper = 2 ** index  # flips the bit at position 'index'
    
    for i in range(dim):
        spin = btest(i, index)
        j = i ^ flipper  # flip the spin at index
        if spin == 1:
            mat[j, i] = 1.0       # ￪ → ￬
        else:
            mat[j, i] = gauge     # ￬ → ￪ (with gauge)
    return mat


def O_dagger(orbitals, index, gauge):
    """Hermitian conjugate of O."""
    return O(orbitals, index, gauge).conj().T


def average_value(operator, eigenvectors, eigenvalues):
    """Computes ⟨ψ₀|O|ψ₀⟩ for the ground state."""
    ground_state = eigenvectors[:, np.argmin(eigenvalues)]
    return np.real(np.dot(np.conjugate(ground_state), np.dot(operator, ground_state)))

def H_slave(h1_up,h1_down, h2_up,h2_down, lamda1, lamda2, U, gauge1_up,gauge1_down, gauge2_up,gauge2_down, J, orbitals):
    """
    h1, h2: h1, h2: Renormalized kinetic energy terms
    lambda1, lambda2: Lagrange multipliers
    U: onsite (intra-orbital) Hubbard repulsion
    U': inter-orbital Hubbard repuslion
    J: Hund's coupling
    gauge1, gauge2: off-diagonal gauge parameters
    """
    dim = 4 ** orbitals
    H = np.zeros((dim, dim), dtype=np.complex128)

    # Indices: 0 (￬1), 1 (￪1), 2 (￬2), 3 (￪2)
    # Lagrange lambda terms
    H += lamda1 * (S_z(orbitals, 0) + S_z(orbitals, 1))
    H += lamda2 * (S_z(orbitals, 2) + S_z(orbitals, 3))

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
    eigvalues, eigvectors = scipy.linalg.eigh(A)
    return eigvalues, eigvectors


def quasiparticle_weight(average):
    """Calculates the quasiparticle weight"""
    return average ** 2

def numerical_gradient(f, x, y, h=1e-5):
    df_dx = (f(x + h, y) - f(x-h, y)) / (2*h)
    df_dy = (f(x, y + h) - f(x, y-h)) / (2*h)
    
    return np.array([df_dx, df_dy])

def gradient_descent(func, start, learn_rate, n_iter=5000, tolerance=1e-8):
    vector = np.array(start, dtype=float)
    for i in range(n_iter):
        x, y = vector
        grad = numerical_gradient(func, x, y)
        if np.all(np.abs(grad) <= tolerance):
            break
        vector -= learn_rate * grad
    return vector


def find_lambda(old_lambda_1, old_lambda_2, U, h1_up,h1_down, h2_up,h2_down, gauge1_up,gauge1_down, gauge2_up,gauge2_down,J,orbitals,nm_1up,nm_1down,nm_2up,nm_2down):
    def spread_func(lamda1,lamda2):
            H = H_slave(h1_up,h1_down, h2_up,h2_down, lamda1, lamda2, U, gauge1_up,gauge1_down, gauge2_up,gauge2_down, J, orbitals)
            evals, eigenvectors = np.linalg.eigh(H)
            average_Sz_1up = average_value(S_z(orbitals,1), eigenvectors, evals)
            spin_occupation_1up = np.real(average_Sz_1up) + 0.5
            average_Sz_1down = average_value(S_z(orbitals,0), eigenvectors, evals)
            spin_occupation_1down = np.real(average_Sz_1down) + 0.5
            
            average_Sz_2up = average_value(S_z(orbitals,3), eigenvectors, evals)
            spin_occupation_2up = np.real(average_Sz_2up) + 0.5
            average_Sz_2down = average_value(S_z(orbitals,2), eigenvectors, evals)
            spin_occupation_2down = np.real(average_Sz_2down) + 0.5

            spread_func_value = (spin_occupation_1up -nm_1up)**2+( spin_occupation_1down - nm_1down) ** 2+(spin_occupation_2up -nm_2up)**2+( spin_occupation_2down - nm_2down) ** 2
            
            return spread_func_value

    start = np.array([old_lambda_1, old_lambda_2])  
    learn_rate = 0.1
    result = gradient_descent(spread_func, start, learn_rate)
    return result[0], result[1]

def dens(m_n, m_c, lamda_tild_1_n,lamda_tild_2_n, Z_1_up,Z_1_down,Z_2_up,Z_2_down):
    ''' This function calculate the density dn/dmu'''
    n_c = fermionic_occupation1(Z_1_up, m_c, lamda_tild_1_n)+fermionic_occupation2(Z_2_up, m_c, lamda_tild_2_n)+fermionic_occupation1(Z_1_down, m_c, lamda_tild_1_n)+fermionic_occupation2(Z_2_down, m_c, lamda_tild_2_n)
    n_n = fermionic_occupation1(Z_1_up, m_n, lamda_tild_1_n)+fermionic_occupation2(Z_2_up, m_n, lamda_tild_2_n)+fermionic_occupation1(Z_1_down, m_n, lamda_tild_1_n)+fermionic_occupation2(Z_2_down, m_n, lamda_tild_2_n)
    if m_c == m_n:
        return 0
    density = (n_c - n_n) / (m_c - m_n)
    if density > 0:
        return max(0.1, density)
    else:
        return min(-0.1, density)


def find_mu(m_c, lamda_tild_1_n,lamda_tild_2_n, Z_1_up,Z_1_down,Z_2_up,Z_2_down,d):
    """Routine to find the chemical potential at each loop."""
    n_c = fermionic_occupation1(Z_1_up, m_c, lamda_tild_1_n)+fermionic_occupation2(Z_2_up, m_c, lamda_tild_2_n)+fermionic_occupation1(Z_1_down, m_c, lamda_tild_1_n)+fermionic_occupation2(Z_2_down, m_c, lamda_tild_2_n)
    return m_c if d==0 else m_c - (n_c - targ_occupation) / d



def Self_consistency_loop(U_values,targ_occupation,Z_1_guess_up,Z_1_guess_down,Z_2_guess_up,Z_2_guess_down,lamda_1_guess,lamda_2_guess,mu_guess,dens_guess,orbitals,tol,filename):
    h_guess1_up=h_guess1_down=0.2
    h_guess2_up=h_guess2_down=0.2
    spin_occupation_1_o_up=spin_occupation_1_o_down=0.5
    spin_occupation_2_o_up=spin_occupation_2_o_down=0.5
    for i,U in enumerate(U_values):
        J=0.0*U
        bool = 0
        iteration=0
        while bool == 0:
            iteration=iteration+1
            
            f_1_up = fermionic_occupation1(Z_1_guess_up, mu_guess, lamda_1_guess)
            f_2_up = fermionic_occupation1(Z_2_guess_up, mu_guess, lamda_2_guess)
            f_1_down = fermionic_occupation1(Z_1_guess_down, mu_guess, lamda_1_guess)
            f_2_down = fermionic_occupation1(Z_2_guess_down, mu_guess, lamda_2_guess)

            gauge1_up = estimate_gauge(f_1_up)
            gauge2_up = estimate_gauge(f_2_up)
            gauge1_down = estimate_gauge(f_1_down)
            gauge2_down = estimate_gauge(f_2_down)
            
            h_new_1_up =calculate_h1( mu_guess, lamda_1_guess, Z_1_guess_up)
            h_new_2_up =calculate_h2( mu_guess, lamda_2_guess, Z_2_guess_up)
            h_new_1_down =calculate_h1( mu_guess, lamda_1_guess, Z_1_guess_down)
            h_new_2_down =calculate_h2( mu_guess, lamda_2_guess, Z_2_guess_down)
                                               
            lamda_1_new, lamda_2_new = find_lambda(lamda_1_guess,lamda_2_guess, U,h_new_1_up,h_new_1_down, h_new_2_up,h_new_2_down, gauge1_up,gauge1_down, gauge2_up,gauge2_down,J,orbitals,f_1_up,f_1_down,f_2_up,f_2_down)
            
            evas, evcs = diagonalizer(H_slave(h_new_1_up,h_new_1_down,h_new_2_up,h_new_2_down, lamda_1_new,lamda_2_new, U, gauge1_up,gauge1_down,gauge2_up,gauge2_down,J,orbitals))

            average_1n_up = average_value(O(orbitals, 1, gauge1_up), evcs, evas)
            Z_1_up = quasiparticle_weight(average_1n_up)
            
            average_2n_up = average_value(O(orbitals, 3, gauge2_up), evcs, evas)
            Z_2_up = quasiparticle_weight(average_2n_up)
            
            average_1n_down = average_value(O(orbitals, 0, gauge1_down), evcs, evas)
            Z_1_down = quasiparticle_weight(average_1n_down)
            
            average_2n_down = average_value(O(orbitals, 2, gauge2_up), evcs, evas)
            Z_2_down = quasiparticle_weight(average_2n_down)
           
            average_Sz_1up = average_value(S_z(orbitals,1), evcs, evas)
            spin_occupation_1up = np.real(average_Sz_1up) + 0.5
            average_Sz_1down = average_value(S_z(orbitals,0), evcs, evas)
            spin_occupation_1down = np.real(average_Sz_1down) + 0.5
            
            average_Sz_2up = average_value(S_z(orbitals,3), evcs, evas)
            spin_occupation_2up = np.real(average_Sz_2up) + 0.5
            average_Sz_2down = average_value(S_z(orbitals,2), evcs, evas)
            spin_occupation_2down = np.real(average_Sz_2down) + 0.5

            mu_new = find_mu(mu_guess, lamda_1_new,lamda_2_new,  Z_1_up,Z_1_down,Z_2_up,Z_2_down,dens_guess)
            #print(mu_new)
            dens_new = dens(mu_new, mu_guess, lamda_1_new,lamda_2_new,   Z_1_up,Z_1_down,Z_2_up,Z_2_down)
            
            
            diff_lamda1 = lamda_1_new - lamda_1_guess
            diff_lamda2 = lamda_2_new - lamda_2_guess
            diff_mu     = mu_new - mu_guess

            diff_h1_up   = h_new_1_up - h_guess1_up
            diff_h2_up   = h_new_2_up - h_guess2_up
            diff_h1_down = h_new_1_down - h_guess1_down
            diff_h2_down = h_new_2_down - h_guess2_down

            diff_Z1_up   = Z_1_up - Z_1_guess_up
            diff_Z2_up   = Z_2_up - Z_2_guess_up
            diff_Z1_down = Z_1_down - Z_1_guess_down
            diff_Z2_down = Z_2_down - Z_2_guess_down
            total_f_occ = f_1_up + f_1_down + f_2_up + f_2_down

            # Convergence check
            if (
            abs(diff_Z1_up)   < tol and abs(diff_Z1_down) < tol and
            abs(diff_Z2_up)   < tol and abs(diff_Z2_down) < tol and
            abs(diff_lamda1)  < tol and abs(diff_lamda2)  < tol and
            abs(diff_h1_up)   < tol and abs(diff_h1_down) < tol and
            abs(diff_h2_up)   < tol and abs(diff_h2_down) < tol and
            abs(diff_mu)      < tol and
            abs(targ_occupation - total_f_occ) < tol and
            abs(spin_occupation_1up - spin_occupation_1_o_up) < tol and
            abs(spin_occupation_2up - spin_occupation_2_o_up) < tol and
            abs(spin_occupation_1down - spin_occupation_1_o_down) < tol and 
            abs(spin_occupation_2down - spin_occupation_2_o_down) < tol
            ):
    
                row=[
                U, mu_new,
                Z_1_up, Z_1_down, Z_2_up, Z_2_down,
                lamda_1_new, lamda_2_new,
                h_new_1_up, h_new_1_down, h_new_2_up, h_new_2_down,
                f_1_up, f_1_down,
                f_2_up, f_2_down,
                spin_occupation_1up, spin_occupation_1down,
                spin_occupation_2up, spin_occupation_2down
                ]
                save_table([row],filename)

                bool = 1
                print("success" + ' U=' + str(U))
                print("success" + ' mu=' + str(mu_new))
            else:
                print('U=' + str(U)+' Iteration ='+str(iteration))
                print('diff_Z1_up       =', abs(diff_Z1_up))
                print('diff_Z1_down     =', abs(diff_Z1_down))
                print('diff_Z2_up       =', abs(diff_Z2_up)) 
                print('diff_Z2_down     =', abs(diff_Z2_down)) 
                print('diff_lamda1      =', abs(diff_lamda1)) 
                print('diff_lamda2      =', abs(diff_lamda2)) 
                print('diff_h1_up       =', abs(diff_h1_up)) 
                print('diff_h1_down     =', abs(diff_h1_down)) 
                print('diff_h2_up       =', abs(diff_h2_up)) 
                print('diff_h2_down     =', abs(diff_h2_down)) 
                print('diff_mu          =', abs(diff_mu)) 
                # Occupation difference 
                print('diff_total_occ   =', abs(targ_occupation - total_f_occ)) 
                # Spin occupations 
                print('diff_spin_1up    =', abs(spin_occupation_1up - spin_occupation_1_o_up)) 
                print('diff_spin_2up    =', abs(spin_occupation_2up - spin_occupation_2_o_up)) 
                print('diff_spin_1down  =', abs(spin_occupation_1down - spin_occupation_1_o_down)) 
                print('diff_spin_2down  =', abs(spin_occupation_2down - spin_occupation_2_o_down))

                
                # Update variables for the next iteration with mixing:
                alpha=0.1
                # Mixing for h 
                h_guess1_up   = alpha * h_new_1_up   + (1 - alpha) * h_guess1_up
                h_guess1_down = alpha * h_new_1_down + (1 - alpha) * h_guess1_down
                h_guess2_up   = alpha * h_new_2_up   + (1 - alpha) * h_guess2_up
                h_guess2_down = alpha * h_new_2_down + (1 - alpha) * h_guess2_down

                # Mixing for spin occupations 
                spin_occupation_1_o_up   = alpha * spin_occupation_1up   + (1 - alpha) * spin_occupation_1_o_up
                spin_occupation_1_o_down = alpha * spin_occupation_1down + (1 - alpha) * spin_occupation_1_o_down
                spin_occupation_2_o_up   = alpha * spin_occupation_2up   + (1 - alpha) * spin_occupation_2_o_up
                spin_occupation_2_o_down = alpha * spin_occupation_2down + (1 - alpha) * spin_occupation_2_o_down

                # Mixing for lambda
                lamda_1_guess = alpha * lamda_1_new + (1 - alpha) * lamda_1_guess
                lamda_2_guess = alpha * lamda_2_new + (1 - alpha) * lamda_2_guess

                # Mixing for Z-factors
                Z_1_guess_up   = alpha * Z_1_up   + (1 - alpha) * Z_1_guess_up
                Z_1_guess_down = alpha * Z_1_down + (1 - alpha) * Z_1_guess_down
                Z_2_guess_up   = alpha * Z_2_up   + (1 - alpha) * Z_2_guess_up
                Z_2_guess_down = alpha * Z_2_down + (1 - alpha) * Z_2_guess_down

                # Mixing for chemical potential and density
                mu_guess   = alpha * mu_new   + (1 - alpha) * mu_guess
                dens_guess = alpha * dens_new + (1 - alpha) * dens_guess

            


def save_table(table, filename):
    header="# U  mu  Z_1_up  Z_1_down  Z_2_up  Z_2_down  lamda_1  lamda_2  h_1_up  h_1_down  h_2_up  h_2_down  f_1_up  f_1_down  f_2_up  f_2_down  spin_1_up  spin_1_down  spin_2_up  spin_2_down  \n"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            file.write(header)
        with open(filename,'a') as file:
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
orbitals=2  
cut_sim=2000#cut off for simpson rule used in fermionic integrals the higher it is the more precise but also the slower it gets
# Define constants
t = 0.5  # Hopping amplitude
D = 2.0 * t  # Half-bandwidth, energy unit
U_values = np.arange(0, 3.5,0.1)# values of the onsite coulomb repulsion
Z_values = np.zeros(len(U_values))  # initialize array to store results of the quasiorbitals weight
tol = 1e-4  # the Tolerance
beta = 1000
targ_occupation = 2
# Initial seed
lamda_1_guess=0
lamda_2_guess=0
mu_guess=0
dens_guess = 1
Z_1_guess_up=Z_1_guess_down=1
Z_2_guess_up=Z_2_guess_down=1

file_path = 'NEW_2_orbitals_bethe_J=U_p=0.dat'

Self_consistency_loop(U_values,targ_occupation,Z_1_guess_up,Z_1_guess_down,Z_2_guess_up,Z_2_guess_down,lamda_1_guess,lamda_2_guess,mu_guess,dens_guess,orbitals,tol,file_path)
