import numpy as np
from random import choice
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class lattice:
    def __init__(self, T, N, J=1, B=0, start='low'):
        self.N = N  # the size of lattice
        self.T = T  # the temperature of out set
        self.J = J  # the coupling coefficient of nearest neighborhood
        self.B = B  # the environmental magnetic field
        if start == 'low':  # constructing from low temperature configuration
            self.lattice = np.ones((self.N, self.N), dtype=int)
        else:
            self.lattice = np.zeros((self.N, self.N), dtype=int)
            for i in range(self.N):
                for j in range(self.N):
                    self.lattice[i, j] = choice([-1, 1])
        self.M_list = [self.M_tot(), ]  # save the total M of every configuration in the equilibrium ensemble
        self.E_list = [self.E_tot(), ]  # save the total E of every configuration in the equilibrium ensemble

    def E_ele(self, i, j):  # only the nearest coupling is considered
        e = 0
        e += -self.J * self.lattice[i, j] * (self.lattice[(i - 1) % self.N, j] + self.lattice[(i + 1) % self.N, j]
                                             + self.lattice[i, (j - 1) % self.N] + self.lattice[i, (j + 1) % self.N])
        e += self.lattice[i, j] * self.B
        return e

    def E_tot(self):
        e = 0
        for i in range(self.N):
            for j in range(self.N):
                e += self.E_ele(i, j)
        return e

    def M_tot(self):
        return np.mean(self.lattice[:])

    # the Markov chain Monte Carlo is constructed by randomly choosing a site and reversing its spin.
    # So, the change of energy and magnetism is the change of those on the site.
    def d_E(self, m, n):
        return -2 * self.E_ele(m, n)

    def Metro(self, num=90000):
        for i in range(num):
            [m, n] = np.random.randint(0, self.N, 2)
            if (self.d_E(m, n) < 0) | (np.random.rand() < np.exp(-self.d_E(m, n) / self.T)):
                self.lattice[m, n] *= -1
                self.E_list.append(self.E_list[-1] + self.d_E(m, n))
                self.M_list.append(self.M_tot())
            else:
                self.E_list.append(self.E_list[-1])
                self.M_list.append(self.M_list[-1])
        return self


T = np.linspace(.2, 8, 20)
M = np.zeros(np.shape(T))
for i in range(np.size(T)):
    if i > np.size(T) / 2:
        st = 'high'
    else:
        st = 'low'
    a = lattice(T[i], 100, start=st)
    a.Metro()
    M[i] = np.mean(a.M_list)
    if i == 0:
        plt.figure()
        plt.imshow(a.lattice)
        plt.title('low temperature')
    if i == np.size(T) - 1:
        plt.figure()
        plt.imshow(a.lattice)
        plt.title('high temperature')
plt.figure()
smooth = interp1d(T, M)
plt.plot(T, M, 'o', T, smooth(T))
plt.title('Magnetism-temperature')
plt.xlabel('temperature')
plt.ylabel('Total Magnetism')
plt.show()