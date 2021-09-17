from .optimizer import Optimizer, Options, cluster
import numpy as np

class PSO_options(Options):
    def __init__(self):
        super().__init__()
        self.n = 40    # population size
        self.w = 0.7
        self.c1 = 2.0
        self.c2 = 2.0

class PSO(Optimizer):
    def __init__(self, M=None):
        super().__init__(M, PSO_options)
        self.name = "PSO"

    @cluster.on_master
    def initialize_state(self):
        # initialize weights
        self.n = 39    # populatin size
        self.w = self.options.w   # initial inertial weight
        self.c1 = self.options.c1  # initial cognitive weight
        self.c2 = self.options.c2  # initial social weight

        # initialize populations
        self.p = np.random.uniform(self.M.p_lb, self.M.p_ub, (self.options.n, self.M.m))  # particle positions
        self.v = np.zeros((self.options.n, self.M.m))        # particle velocities
        self.pbest_p = np.zeros((self.options.n, self.M.m))  # personal best positions
        self.y = np.full(self.options.n, np.inf)             # particle actual values
        self.pbest_y = np.full(self.options.n, np.inf)       # particle personal best values

    @cluster.on_master
    def update_pbest(self):
        idxs = np.where(self.y < self.pbest_y)[0]
        self.improved = idxs.size
        self.pbest_y[idxs] = self.y[idxs]
        self.pbest_p[idxs, :] = self.p[idxs, :]

    @cluster.on_master
    def update_particle_positions(self):
        r1 = np.random.random(self.options.n)[:,None]
        r2 = np.random.random(self.options.n)[:,None]
        self.v *= self.w                              # momentum component
        self.v += self.c1*r1*(self.pbest_p - self.p)  # cognitive component
        self.v += self.c2*r2*(self.M.p - self.p)      # social component
        self.p += self.v

    @cluster.on_master
    def generate_new_epoch_data(self):
        self.update_pbest()
        self.update_particle_positions()

