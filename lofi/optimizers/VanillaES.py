from .optimizer import Optimizer, Options, cluster, np

class VanillaES_options(Options):
    def __init__(self):
        super().__init__()
        self.n = 19
        self.lr = 0.01
        self.sigma = 1e-2

class VanillaES(Optimizer):
    def __init__(self, M=None):
        super().__init__(M, VanillaES_options)

    @cluster.on_master
    def noise(self):
        return np.random.normal(0, 1, (self.options.n, self.M.m))

    @cluster.on_master
    def initialize_state(self):
        self.iter = 0

        # initial population allocation
        self.p = np.zeros((2*self.options.n, self.M.m))
        self.y = np.full(2*self.options.n, np.inf)

    @cluster.on_master
    def generate_new_samples(self):
        self.epsilon = self.noise()
        self.delta = self.options.sigma*self.epsilon
        self.p[:self.options.n] = self.M.p + self.delta
        self.p[self.options.n:] = self.M.p - self.delta

    @cluster.on_master
    def grad_estimation(self):
        y_pos = self.y[:self.options.n]
        y_neg = self.y[self.options.n:]
        grad = self.epsilon.transpose().dot(y_pos - y_neg)/self.options.n
        return grad

    @cluster.on_master
    def update_model(self):
        self.M.p = self.M.p - self.options.lr*self.grad_estimation()
