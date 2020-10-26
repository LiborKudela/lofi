from .PSO import PSO, np, cluster

class GRAPSO(PSO):
    def __init__(self, M=None):
        super().__init__(M)
        if cluster.global_rank == 0:
            self.name = "GRAPSO"

    def adapt_swarm_weights(self):
        # update swarm weights
        if np.random.random() > 0.5:
            self.w = 0.4 + 0.5*np.random.random()
        if np.random.random() > 0.5:
            self.c1 = 1.5 + np.random.random()
        if np.random.random() > 0.5:
            self.c2 = 1.5 + np.random.random()
