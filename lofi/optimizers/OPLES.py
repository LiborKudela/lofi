from .optimizer import Optimizer, Options, cluster, np

class OPLES_options(Options):
    def __init__(self):
        super().__init__()
        self.max_sigma = 1e-1        # max sampling deviation
        self.min_sigma = 1e-8        # min sampling deviation
        self.sigma_init = 1e-2       # initial sigma
        self.success_rule = 0.11     # this seems to work OK

class OPLES(Optimizer):
    def __init__(self, M=None):
        super().__init__(M, OPLES_options)
        self.name = "OPLES"

    @cluster.on_master
    def initialize_state(self):
        self.sigma = 0.01
        self.success_rate = 0
        self.Ldrop_rate = 0                 # loss decrease rate
        self.C_rate = 0                     # convergence rate
        self.iter = 0                       # iteration counter
        self.velocity = np.zeros(self.M.m)  # parental velocity
        self.velocity_norm = 0              # parental velocity norm

        #initialize population
        self.z = np.random.normal(0, 1, (self.options.n, self.M.m))
        self.p = self.M.p + self.sigma*self.z
        self.y = np.full(self.options.n, np.inf)

    @cluster.on_master
    def pre_gbest_update_actions(self):
        self.prev_gbest_y = self.M.y
        self.prev_gbest_p = np.copy(self.M.p)

    @cluster.on_master
    def adaptation(self):
        #TODO: clean me please
        self.success_rate = 0.8*self.success_rate + 0.2*np.count_nonzero(self.y <= self.prev_gbest_y)/self.options.n
        self.velocity[:] = 0.9*self.velocity + 0.1*(self.M.p-self.prev_gbest_p)
        self.velocity_norm = np.linalg.norm(self.velocity)
        if self.velocity_norm > 1e-7:
            self.velocity[:] = 0.9*self.velocity + 0.1*(self.M.p-self.prev_gbest_p)
        else:
            self.velocity[:] = self.M.p-self.prev_gbest_p
        self.velocity_norm = np.linalg.norm(self.velocity)
        self.sigma = 0.9*self.sigma + 0.1*self.sigma*np.exp(self.success_rate - self.options.success_rule)
        self.sigma = np.clip(self.sigma, self.options.min_sigma, self.options.max_sigma)
        self.C_rate = 0.9*self.C_rate + 0.1*(1-self.M.y/self.prev_gbest_y)
        if self.C_rate < 1e-7 and self.sigma == self.options.min_sigma:
            self.sigma = 10000*self.sigma
        self.Ldrop_rate = min(self.M.y, 0.9*self.Ldrop_rate + 0.1*(self.prev_gbest_y - self.M.y))

    @cluster.on_master
    def generate_new_epoch_data(self):
        self.z = np.random.normal(0, 1, (self.options.n, self.M.m))
        self.p = self.M.p + self.sigma*self.z + self.velocity
        if self.velocity_norm > 1e-7:
            self.p[0] = self.M.p + 0.9*self.velocity
            self.p[1] = self.M.p + 1.1*self.velocity
