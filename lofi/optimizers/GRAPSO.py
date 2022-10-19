from .PSO import PSO, np, cluster

class GRAPSO(PSO):
    def __init__(self, M=None, n=40, w=0.7, c1=2.0, c2=2.0, bound_control=True,
                 sparse=False, sparsity_weight=1.0):
                 
        super().__init__(M=M, n=n, w=w, c1=c1, c2=c2, 
                         bound_control=bound_control, sparse=sparse,
                         sparsity_weight=sparsity_weight)

    @cluster.on_master
    def update_model(self):
        # update model
        best_idx = np.nanargmin(self.y)
        if self.M.y > self.y[best_idx]:
            self.M.p = self.p[best_idx,:]
        else:
            # adapt swarm weights
            if np.random.random() > 0.5:
                self.w = 0.3 + 0.6*np.random.random()
            if np.random.random() > 0.5:
                self.c1 = 1.5 + np.random.random()
            if np.random.random() > 0.5:
                self.c2 = 1.5 + np.random.random()
