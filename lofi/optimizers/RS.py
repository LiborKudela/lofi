from .optimizer import Optimizer, cluster, np


class RS(Optimizer):
    def __init__(self, M=None, n=19, bound_control=True, sparse=False, 
                 sparsity_weight=1.0):

        self.n = n

        super().__init__(M=M, bound_control=bound_control, sparse=sparse, 
                         sparsity_weight=sparsity_weight)

    @cluster.on_master
    def initialize_state(self):
        self.sigma = 1e-5
        self.iter = 0                       # iteration counter

        #initialize population
        self.z = np.random.normal(0, 1, (self.n, self.M.m))
        self.p = self.M.p + self.sigma*self.z
        self.y = np.full(self.n, np.inf)

    @cluster.on_master
    def generate_new_samples(self):
        self.z = np.random.normal(0, 1, (self.n, self.M.m))
        self.p = self.M.p + self.sigma*self.z

    @cluster.on_master
    def update_model(self):
        best_idx = np.nanargmin(self.y)
        if self.M.y > self.y[best_idx]:
            self.M.p = self.p[best_idx,:]

