from .optimizer import Optimizer, cluster, np

class PSO(Optimizer):
    def __init__(self, M=None, n=40, w=0.7, c1=2.0, c2=2.0, bound_control=True,
                 sparse=False, sparsity_weight=1.0):

        self.n = n    # population size
        self.w = w    # inertia weight
        self.c1 = c1  # congnitive weight
        self.c2 = c2  # social weight

        super().__init__(M=M, bound_control=bound_control, sparse=sparse,
                         sparsity_weight=sparsity_weight)

    @cluster.on_master
    def initialize_state(self):
        # initialize populations
        self.p = np.random.uniform(self.M.p_lb, self.M.p_ub, (self.n, self.M.m))  # particle positions
        self.v = np.zeros((self.n, self.M.m))        # particle velocities
        self.pbest_p = np.zeros((self.n, self.M.m))  # personal best positions
        self.y = np.full(self.n, np.inf)             # particle actual values
        self.pbest_y = np.full(self.n, np.inf)       # particle personal best values

    @cluster.on_master
    def update_pbest(self):
        idxs = np.where(self.y < self.pbest_y)[0]
        self.improved = idxs.size
        self.pbest_y[idxs] = self.y[idxs]
        self.pbest_p[idxs, :] = self.p[idxs, :]

    @cluster.on_master
    def update_particle_positions(self):
        r1 = np.random.random(self.n)[:,None]
        r2 = np.random.random(self.n)[:,None]
        self.v *= self.w                              # momentum component
        self.v += self.c1*r1*(self.pbest_p - self.p)  # cognitive component
        self.v += self.c2*r2*(self.M.p - self.p)      # social component
        self.p += self.v

    @cluster.on_master
    def generate_new_samples(self):
        self.update_pbest()
        self.update_particle_positions()

    @cluster.on_master
    def update_model(self):
        best_idx = np.nanargmin(self.y)
        if self.M.y > self.y[best_idx]:
            self.M.p = self.p[best_idx,:]

