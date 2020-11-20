from .PSO import PSO, np, cluster

class GRAPSO(PSO):
    def __init__(self, M=None):
        super().__init__(M)
        if cluster.global_rank == 0:
            self.name = "GRAPSO"
            
    @cluster.on_master
    def adaptation(self):
        # update swarm weights
        if np.random.random() > 0.5:
            self.w = 0.3 + 0.6*np.random.random()
        if np.random.random() > 0.5:
            self.c1 = 1.5 + np.random.random()
        if np.random.random() > 0.5:
            self.c2 = 1.5 + np.random.random()
