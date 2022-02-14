from .VanillaES import VanillaES, VanillaES_options, cluster, np

class RMSPropES_options(VanillaES_options):
    def __init__(self):
        super().__init__()

class RMSPropES(VanillaES):
    def __init__(self, M=None, sigma=1e-6, lr=0.01, alpha=0.99, eps=1e-8):
        super().__init__(M)
        self.S_grad =  np.zeros(self.M.m)
        self.sigma = sigma
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

    @cluster.on_master
    def update_model(self):
        grad = self.grad_estimation()

        # Exponential averaging of squared gradient
        self.S_grad *= self.alpha
        self.S_grad += (1-self.alpha)*grad**2

        # parameter update
        self.M.p = self.M.p - self.lr*(grad/(np.sqrt(self.S_grad)+self.eps))

