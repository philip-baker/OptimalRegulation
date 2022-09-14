# Complementary Test Code to the dissertation "On Optimal Regulation" -  Appendix A - Section A.1 Numerical Simulations
# Code Function: Reproduce Figure 2.1 and Figure 2.2 in the dissertation
# Author: Philip Baker
# Date: September 2022
# Contact: philip.baker@warwick.ac.uk



from scipy import optimize as op
import numpy as np
import matplotlib.pyplot as plt

# USER INPUTS
N = 20  # Number of iterations
Bank_1 = 110  # Total assets (incl. capital buffer) of Bank 1
Bank_2 = 110  # Total assets (incl. capital buffer) of Bank 1
C = 5202000  # Max allowable systemic risk
a_11 = 55
a_12 = 45
a_21 = 45
a_22 = 55

# hard coded capital requirement functinos and corresponding derivatives
def z1_req(a11, a12, a21, a22, C):
    return ((a11 + a11 * a21 + a12 * a22 + a12) + (a21 + a11 * a21 + a12 * a22 + a22)) * (
                a11 + a11 * a21 + a12 * a22 + a12) / C


def z2_req(a11, a12, a21, a22, C):
    return ((a11 + a11 * a21 + a12 * a22 + a12) + (a21 + a11 * a21 + a12 * a22 + a22)) * (
                a21 + a11 * a21 + a12 * a22 + a22) / C


def V_a11(a11, a12, a21, a22, C):
    return (a22 + a21 ** 2 * (1 + 4 * a11) + 3 * a22 * a12 + 2 * (a11 + a21) + a21 * (
                1 + a22 + 6 * a12 + 3 * a21 + 4 * a22 * a12)) / C


def V_a12(a11, a12, a21, a22, C):
    # using symmetry
    return V_a11(a12, a11, a22, a21, C)


def V_a21(a11, a12, a21, a22, C):
    return V_a11(a21, a22, a11, a12, C)


def V_a22(a11, a12, a21, a22, C):
    # using symmetry
    return V_a12(a21, a22, a11, a12, C)


def simulate_Directional_Model(N, Bank_1, Bank_2, C, a_11, a_12, a_21, a_22):
    # preallocate arrays
    holdings = np.zeros((4, N))
    req_z = np.zeros((2, N))
    z = np.zeros((2, N))
    V = np.zeros((4, N))
    np.shape(holdings)

    holdings[0, 0] = a_11
    holdings[1, 0] = a_12
    holdings[2, 0] = a_21
    holdings[3, 0] = a_22

    # initial capital holding
    z[0, 0] = Bank_1 - (holdings[0, 0] + holdings[1, 0])
    z[1, 0] = Bank_2 - (holdings[2, 0] + holdings[3, 0])

    def objective_function(a):
        gamma_1 = 1
        gamma_2 = 1
        return - gamma_1 * a[0] - gamma_2 * a[1]

    for i in range(0, N - 1):
        req_z[0, i] = z1_req(holdings[0, i], holdings[1, i], holdings[2, i], holdings[3, i], C)
        req_z[1, i] = z2_req(holdings[0, i], holdings[1, i], holdings[2, i], holdings[3, i], C)
        V[0, i] = V_a11(holdings[0, i], holdings[1, i], holdings[2, i], holdings[3, i], C)
        V[1, i] = V_a12(holdings[0, i], holdings[1, i], holdings[2, i], holdings[3, i], C)
        V[2, i] = V_a21(holdings[0, i], holdings[1, i], holdings[2, i], holdings[3, i], C)
        V[3, i] = V_a22(holdings[0, i], holdings[1, i], holdings[2, i], holdings[3, i], C)

        # Optimise for bank 1 (Bank 1's reaction)
        # capital requirements constrinat
        def constraint_one(a):
            return a[2] - (a[0] - holdings[0, i]) * V[0, i] - (a[1] - holdings[1, i]) * V[1, i] - req_z[0, i]

        # total value constraint
        def constraint_two(a):
            return a[0] + a[1] + a[2] - Bank_1

        con1 = {'type': 'ineq', 'fun': constraint_one}
        con2 = {'type': 'eq', 'fun': constraint_two}
        cons = (con1, con2)

        result = op.minimize(fun=objective_function, method='SLSQP', x0=[holdings[0, i], holdings[1, i], z[0, i]],
                             bounds=((0, Bank_1), (0, Bank_1), (0, Bank_1)), constraints=cons)
        holdings[0, i + 1] = result['x'][0]
        holdings[1, i + 1] = result['x'][1]

        # Optimise for bank 2 (Bank 2's reaction)
        def constraint_one(a):
            return a[2] - (a[0] - holdings[2, i]) * V[2, i] - (a[1] - holdings[3, i]) * V[3, i] - req_z[1, i]

        # total value constraint
        def constraint_two(a):
            return a[0] + a[1] + a[2] - Bank_2

        con1 = {'type': 'ineq', 'fun': constraint_one}
        con2 = {'type': 'eq', 'fun': constraint_two}
        cons = (con1, con2)

        result = op.minimize(fun=objective_function, method='SLSQP', x0=[holdings[2, i], holdings[3, i], z[1, i]],
                             bounds=((0, Bank_2), (0, Bank_2), (0, Bank_2)), constraints=cons)
        holdings[2, i + 1] = result['x'][0]
        holdings[3, i + 1] = result['x'][1]
    return holdings


# Optimal Solution
a_11 = 55
a_12 = 45
a_21 = 45
a_22 = 55
holdings = simulate_Directional_Model(N, Bank_1, Bank_2, C, a_11, a_12, a_21, a_22)

holdings = np.round(holdings, 2)  # round holdings to two decimal places for formatting purposes

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle("Directional Model - Optimal Solution", fontsize=14)
p1, = ax1.plot(holdings[0, :], label="Asset 1")
p2, = ax1.plot(holdings[1, :], label="Asset 2")
ax1.set_title("Bank 1 Asset Holdings", fontsize=14)
ax1.set_ylabel("Units of Numeraire", fontsize=14)
ax1.set_xlabel("Iteration", fontsize=14)
ax1.legend(loc='upper left', fontsize=14)

p3, = ax2.plot(holdings[2, :], label="Asset 1")
p4, = ax2.plot(holdings[3, :], label="Asset 2")
ax2.set_title("Bank 2 Asset Holdings", fontsize=14)
ax2.set_ylabel("Units of Numeraire", fontsize=14)
ax2.set_xlabel("Iteration", fontsize=14)
ax2.legend(loc='upper left', fontsize=14)

plt.savefig("figs/DirectionalModelOptimalSolution.png")
plt.show()

# Periodic Solution
a_11 = 50
a_12 = 45
a_21 = 50
a_22 = 45

holdings = simulate_Directional_Model(N, Bank_1, Bank_2, C, a_11, a_12, a_21, a_22)

holdings = np.round(holdings, 2)  # round holdings to two decimal places for formatting purposes

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle("Directional Model - Periodic Solution", fontsize=14)
p1, = ax1.plot(holdings[0, :], label="Asset 1")
p2, = ax1.plot(holdings[1, :], label="Asset 2")
ax1.set_title("Bank 1 Asset Holdings", fontsize=14)
ax1.set_ylabel("Units of Numeraire", fontsize=14)
ax1.set_xlabel("Iteration", fontsize=14)
ax1.legend(loc='upper left', fontsize=14)

p3, = ax2.plot(holdings[2, :], label="Asset 1")
p4, = ax2.plot(holdings[3, :], label="Asset 2")
ax2.set_title("Bank 2 Asset Holdings", fontsize=14)
ax2.set_ylabel("Units of Numeraire", fontsize=14)
ax2.set_xlabel("Iteration", fontsize=14)
ax2.legend(loc='upper left', fontsize=14)
plt.savefig("figs/DirectionalModelPeriodicSolution.png")
plt.show()

# Non-Optimal Steady State
a_11 = 50
a_12 = 50
a_21 = 50
a_22 = 50

holdings = simulate_Directional_Model(N, Bank_1, Bank_2, C, a_11, a_12, a_21, a_22)

holdings = np.round(holdings, 2)  # round holdings to two decimal places for formatting purposes

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle("Directional Model - Inefficient Equilibrium", fontsize=14)
p1, = ax1.plot(holdings[0, :], label="Asset 1")
p2, = ax1.plot(holdings[1, :], label="Asset 2")
ax1.set_title("Bank 1 Asset Holdings", fontsize=14)
ax1.set_ylabel("Units of Numeraire", fontsize=14)
ax1.set_xlabel("Iteration", fontsize=14)
ax1.legend(loc='upper left', fontsize=14)

p3, = ax2.plot(holdings[2, :], label="Asset 1")
p4, = ax2.plot(holdings[3, :], label="Asset 2")
ax2.set_title("Bank 2 Asset Holdings", fontsize=14)
ax2.set_ylabel("Units of Numeraire", fontsize=14)
ax2.set_xlabel("Iteration", fontsize=14)
ax2.legend(loc='upper left', fontsize=14)
plt.savefig("figs/DirectionalModelInefficientEquilibrium.png")
plt.show()

# Optimal Solution
a_11 = 12
a_12 = 11
a_21 = 11
a_22 = 12
holdings = simulate_Directional_Model(N, Bank_1, Bank_2, C, a_11, a_12, a_21, a_22)

holdings = np.round(holdings, 2)  # round holdings to two decimal places for formatting purposes

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle("Directional Model - Optimal Solution", fontsize=14)
p1, = ax1.plot(holdings[0, :], label="Asset 1")
p2, = ax1.plot(holdings[1, :], label="Asset 2")
ax1.set_title("Bank 1 Asset Holdings", fontsize=14)
ax1.set_ylabel("Units of Numeraire", fontsize=14)
ax1.set_xlabel("Iteration", fontsize=14)
ax1.legend(loc='upper left', fontsize=14)

p3, = ax2.plot(holdings[2, :], label="Asset 1")
p4, = ax2.plot(holdings[3, :], label="Asset 2")
ax2.set_title("Bank 2 Asset Holdings", fontsize=14)
ax2.set_ylabel("Units of Numeraire", fontsize=14)
ax2.set_xlabel("Iteration", fontsize=14)
ax2.legend(loc='upper left', fontsize=14)
plt.savefig("figs/DirectionalModelOptimalEquilibrium.png")
plt.show()