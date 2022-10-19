from .optimizer import Optimizer, cluster, np

class RMSPropES(Optimizer):
    def __init__(self, M=None, n=5, lr=0.01, sigma=1e-6, alpha=0.99, eps=1e-8,
                 bound_control=True, sparse=False, sparsity_weight=1.0):

        self.n = n
        self.sigma = sigma
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

        super().__init__(M=M, bound_control=bound_control, sparse=sparse, 
                         sparsity_weight=sparsity_weight)

    @cluster.on_master
    def noise(self):
        r = np.random.normal(0, 1, (self.n, self.M.m))
        return r

    @cluster.on_master
    def initialize_state(self):
        self.iter = 0
        
        # initial population allocation
        self.S_grad =  np.zeros(self.M.m)
        self.p = np.zeros((2*self.n, self.M.m))
        self.y = np.full(2*self.n, np.inf)

    @cluster.on_master
    def generate_new_samples(self):
        self.epsilon = self.noise()
        self.delta = self.sigma*self.epsilon
        self.p[:self.n] = self.M.p + self.delta
        self.p[self.n:] = self.M.p - self.delta

    @cluster.on_master
    def grad_estimation(self):
        y_pos = self.y[:self.n]
        y_neg = self.y[self.n:]
        grad = self.epsilon.transpose().dot(y_pos - y_neg)/self.n
        return grad

    @cluster.on_master
    def update_model(self):
        grad = self.grad_estimation()

        # Exponential averaging of squared gradient
        self.S_grad *= self.alpha
        self.S_grad += (1-self.alpha)*grad**2

        # parameter update
        self.M.p = self.M.p - self.lr*(grad/((np.sqrt(self.S_grad)+self.eps)))

