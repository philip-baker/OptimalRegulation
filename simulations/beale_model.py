# Complementary Test Code to the dissertation "On Optimal Regulation" -  Chapter 2 - Section 1.5 Beale's Approach
# Code Function: Reproduce Figure 2.1 and Figure 2.2 in the dissertation
# Author: Philip Baker
# Date: September 2022
# Contact: philip.baker@warwick.ac.uk

import numpy as np
import matplotlib.pyplot as plt


def compute_holdings(B, C, y):
    '''Simulates "Beale's Approach" to financial regulation (Section 1.5), in particular the case presented
       in the dissertation as a counterexample to Beale's Conjecture. Assuming that banks have identical asset
       allocations, and equal amounts in each asset this function returns the optimal holdings each bank can take.
       The systemic risk function is C(z,A) = (a_11+a_{11}a_{21}+a_{12}a_{22}+a_{12})^2/z_1 +
       (a_21+a_{11}a_{21}+a_{12}a_{22}+a_{22})^2/z_2
    Inputs:
        B (float) Value of Bank i (B_i)
        C (float) Maximal systemic risk C^*
        y (float) Holdings of each bank (a)
    '''

    # compute the optimal holding for bank i
    x = (np.sqrt(4 * B * C * (2 * y ** 2 + 3 * y + 1) + (C + 2 * y * (y + 1)) ** 2) - C - 2 * y ** 2 - 2 * y) / (
            8 * y ** 2 + 12 * y + 4)
    return x


# set the initial conditions
true_holdings = np.zeros(25)
true_holdings[0] = 50

# simulate Beale's approach
for i in range(0, 24):
    true_holdings[i + 1] = compute_holdings(110, 2 * 10 ** 6, true_holdings[i])

# plot Figure 2.1
plt.plot(true_holdings)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("a", fontsize=14)
plt.savefig("figs/Beale_Model_Progression.png")

# plot figure 2.2. a visualisation of the systemic risk constraint
a = np.linspace(-100, 50, 1000)

# systemic risk as a function of holdings
systemic_risk = (8 * a ** 4 + 16 * a ** 3 + 4 * a ** 2) / (110 - 2 * a)

# systemic risk limit
C = np.zeros(1000) + 2 * 10 ** 6

# plot figure 2.2
plt.plot(a, systemic_risk)
plt.plot(a, C)
plt.xlabel("a", fontsize=14)
plt.ylabel("Systemic Risk (C)", fontsize=14)
plt.savefig("figs/CounterexampleHoldings.png")
