import numpy as np
import matplotlib.pyplot as plt
from chemistrylab.ode_algorithms.spectra import diff_spectra as spec

# Reactions
# 1) A + B -> C
# 2) A + B -> D
# 3) D -> |E|

R = 0.008314462618 # Gas constant (kPa * m**3 * mol**-1 * K**-1)
# constant for k = e^(-E/RT) for each reaction
# 1) A + B -> C
A1 = 400.0
E1 = 50.0
# 2) A + B -> D
A2 = 0.5
E2 = 0.5
# 3) D -> |E|
A3 = 1.0
E3 = 1.0

class Reaction(object):
    def __init__(self, overlap = False):
        self.initial_in_hand = np.array([1.0, 1.0]) # Amount of each reactant in hand (mols) avaliable
        self.nmax = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) # The maximum of each chemical (mols)
        self.labels = ['[A]', '[B]', '[C]', '[D]', '[E]']
        self.rate = np.zeros(3) # rate of each reaction
        self.max_mol = 2.0
        # Parameters to generate up to 3 Gaussian peaks per species
        self.params = []
        if overlap:
            # overlap spectra
            self.params.append(spec.S_6) # spectra for A
            self.params.append(spec.S_7) # spectra for B
            self.params.append(spec.S_8) # spectra for C
            self.params.append(spec.S_3_3) # spectra for D
        else:
            # no overlap spectra
            self.params.append(spec.S_6) # spectra for A
            self.params.append(spec.S_7) # spectra for B
            self.params.append(spec.S_8) # spectra for C
            self.params.append(spec.S_9) # spectra for D
            self.params.append(spec.S_9) # spectra for E
        self.reset()

    def get_ni_label(self):
        return ['[A]', '[B]']

    def get_ni_num(self):
        num_list = []
        for i in range(self.cur_in_hand.shape[0]):
            num_list.append(self.cur_in_hand[i])
        return num_list

    # Reinitializes the reaction for reset purpose in main function
    def reset(self):
        self.cur_in_hand = 1.0 * self.initial_in_hand 
        self.n = np.zeros(self.nmax.shape[0], dtype=np.float32)
        #n[0] is A, n[1] is B, n[2] is C ......

    # update the concentration of each chemical for a time step under tempurature T
    def update(self, T, V, dt):
        C = self.get_concentration(V)
        dC = np.zeros(self.n.shape[0]) # change of concentration of A B C D E
        k1 = A1 * np.exp( (-1 * E1) / (R * T))
        k2 = A2 * np.exp( (-1 * E2) / (R * T))
        k3 = A3 * np.exp( (-1 * E3) / (R * T))
        self.rate[0] = k1 * C[0] * C[1] * dt # Rate of reaction A + B -> C
        self.rate[1] = k2 * C[0] * C[1] * dt # Rate of reaction A + B -> D
        self.rate[2] = k3 * C[3] * dt # Rate of reaction D -> |E|
        dC[0] = -1.0 * self.rate[0] - self.rate[1] # change of A
        dC[1] = -1.0 * self.rate[0] - self.rate[1] # change of B
        dC[2] = self.rate[0] # change of C
        dC[3] = self.rate[1] - self.rate[2] # change of D
        dC[4] = self.rate[2] # change of E
        # Update concentration of each chemical
        for i in range(self.n.shape[0]):
            self.n[i] += dC[i] * V * 1000 # need convert V from m^3 to L
        d_reward = dC[3] * V * 1000 # Reward is change in concentration of D
        return d_reward

    def get_total_pressure(self, V, T = 300):
        P_total = 0
        for i in range(self.n.shape[0]):
            P_total += self.n[i] * R * T / V
        return P_total

    def get_part_pressure(self, V, T = 300):
        P = np.zeros(self.n.shape[0], dtype=np.float32)
        for i in range(self.n.shape[0]):
            P[i] = self.n[i] * R * T / V
        return P

    def get_concentration(self, V = 0.1):
        C = np.zeros(self.n.shape[0], dtype=np.float32)
        for i in range(self.n.shape[0]):
            C[i] = self.n[i] / ( V * 1000)
        return C

    def get_spectra(self, V):
        # Initialize array for wavelength[0, 1] and absorbance
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
        absorb = np.zeros(x.shape[0], dtype=np.float32)
        C = self.get_concentration(V)
        for i in range(len(self.params)): # for each species
            for j in range(self.params[i].shape[0]): # for each peak 
                for k in range(x.shape[0]):
                    absorb[k] += C[i] * self.params[i][j, 0] * np.exp(-0.5 * ((x[k] - self.params[i][j, 1]) / self.params[i][j, 2]) ** 2.0)
                                # amount    *    height      *  decay rate
        # Maximum possible absorbance at any wavelength is 1.0
        absorb = np.clip(absorb, 0.0, 1.0)
        return absorb

    def get_spectra_peak(self, V):
        spectra_peak = []
        C = self.get_concentration(V)
        spectra_peak.append([self.params[0][:, 1] * 600 + 200, C[0] * self.params[0][:, 0], 'A'])
        spectra_peak.append([self.params[1][:, 1] * 600 + 200, C[1] * self.params[1][:, 0], 'B'])
        spectra_peak.append([self.params[2][:, 1] * 600 + 200, C[2] * self.params[2][:, 0], 'C'])
        spectra_peak.append([self.params[3][:, 1] * 600 + 200, C[3] * self.params[3][:, 0], 'D'])
        spectra_peak.append([self.params[4][:, 1] * 600 + 200, C[4] * self.params[4][:, 0], 'E'])
        return spectra_peak

    def get_dash_line_spectra(self, V):
        dash_spectra = []
        C = self.get_concentration(V)
        x = np.linspace(0, 1, 200, endpoint=True, dtype=np.float32)
        for i in range(len(self.params)): # for each species
            each_absorb = np.zeros(x.shape[0], dtype=np.float32)
            for j in range(self.params[i].shape[0]): # for each peak 
                for k in range(x.shape[0]):
                    each_absorb[k] += C[i] * self.params[i][j, 0] * np.exp(-0.5 * ((x[k] - self.params[i][j, 1]) / self.params[i][j, 2]) ** 2.0)
                                # amount    *    height      *  decay rate
            dash_spectra.append(each_absorb)
        return dash_spectra

    def plot_graph(self):
        A = 1.0 # Initial A
        B = 1.0 # Initial B
        C = 0.0 # Initial C
        D = 0.0 # Initial D
        E = 0.0 # Initial D

        T = 300.0 # Temperature (K)
        P = 101.325 # Pressure (kPa)
        dt = 0.01 # Time step (s)

        k1 = T / 300.0 # Rate of reaction 1
        k2 = T / 400.0 # Rate of reaction 2
        k3 = P / 101.325 # Rate of reaction 3

        n_steps = 500 # Number of steps to evolve system
        t = np.zeros(n_steps, dtype=np.float32) # Keep track of time for plotting
        conc = np.zeros((n_steps, 4), dtype=np.float32) # Keep track of concentrations for plotting
        conc[0, 0] = 1.0 * A
        conc[0, 1] = 1.0 * B
        conc[0, 2] = 1.0 * C
        conc[0, 3] = 1.0 * D

        for i in range(1, n_steps):
            # Calculate rate for this time step
            r1 = k1 * A * B
            r2 = k2 * A * B 
            r3 = k3 * D

            # Update concentrations
            A += (-1.0 * r1 - r2) * dt
            B += (-1.0 * r1 - r2) * dt
            C += r1 * dt
            D += (r2 - r3) * dt
            E += r3 * dt

            # Update plotting info
            conc[i, 0] = 1.0 * A
            conc[i, 1] = 1.0 * B
            conc[i, 2] = 1.0 * C
            conc[i, 3] = 1.0 * D
            conc[i, 4] = 1.0 * E
            t[i] = t[i-1] + dt

        plt.figure()
        for i in range(conc.shape[1]):
            plt.plot(t, conc[:, i], label=self.labels[i])
        plt.xlim([t[0], t[-1]])
        plt.ylim([0.0, 2.0])
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration (M)')
        plt.legend()
        plt.savefig('reaction_ex_4.pdf')
        plt.show()
        plt.close()
