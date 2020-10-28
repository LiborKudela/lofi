import numpy as np
from ..cluster import cluster
import time
import matplotlib.pyplot as plt

class options():
    def __init__(self):
        self.n = 40                  # populations size
        self.init_w = 0.7            # initial inertia weight
        self.init_c1 = 2.0           # initial social weight
        self.init_c2 = 2.0           # initial cognitive weight
        self.max_iter = 0            # maximum number of iterations
        self.max_error = -np.inf     # maximum allowed error
        self.limit_space = True      # optional coordinate bound control
        self.sparse_program = False  # optional regularization for sparsity
        self.sparsity_weight = 1.0   # weight to penalize density of program
        self.active_callback = True  # activates reports to terminal

class PSO():
    def __init__(self, M=None):
        self.options = options()
        self.callback = callback
        self.name = "PSO"
        if M is not None:
            self.connect_model(M)
            self.restart()

    def connect_model(self, M):
        self.M = M  # reference to model instance

    def disconnect_model(self):
        self.M = None

    def restart(self):
        self.terminate = False  # termination condition state
        self.epoch_time = 0.0   # time elapsed during last epoch
        self.total_time = 0.0   # total elapsed time during training
        self.options.max_iter = 0

        if cluster.global_rank == 0:
            self.w = self.options.init_w    # initial inertial weight
            self.c1 = self.options.init_c1  # initial cognitive weight
            self.c2 = self.options.init_c2  # initial social weight
            self.iter = 0                   # iteration counter

            # initialize populations
            self.p = np.random.uniform(self.M.p_lb, self.M.p_ub, (self.options.n, self.M.m))  # particle positions
            self.v = np.zeros((self.options.n, self.M.m))        # particle velocities
            self.pbest_p = np.zeros((self.options.n, self.M.m))  # personal best positions
            self.y = np.full(self.options.n, np.inf)           # particle values
            self.pbest_y = np.full(self.options.n, np.inf)     # particle personal best values
        else:
            self.p = None  # this is necessary for cluster.map call

    def extract_results(self, r):
        if cluster.global_rank == 0:
            # mutate r into an array so each value type corresponds to a single row
            # r[0,:]=losses, r[1,:]=densities, r[2,:]=return codes, r[3,:]=sim. elapsed times
            r = np.array(r).transpose()

            # read results by type (rows)
            self.y[:] = r[0,:]
            if self.options.sparse_program:
                self.y += r[1,:]
            self.failed = np.count_nonzero(r[2,:])
            self.survived = self.options.n - self.failed
            self.mean_sim_cpu_time = np.sum(r[3,:])/self.options.n
            self.result_owner = r[4,:]

    def update_pbest(self):
        if cluster.global_rank == 0:
            better = np.where(self.y < self.pbest_y)[0]
            self.improved = better.size
            self.pbest_y[better] = self.y[better]
            self.pbest_p[better, :] = self.p[better, :]

    def update_gbest(self):
        if cluster.global_rank == 0:
            best_idx = np.nanargmin(self.pbest_y)
            if self.M.y > self.pbest_y[best_idx]:
                self.M.y = self.pbest_y[best_idx]
                self.M.p[:] = self.p[best_idx,:]
                self.M.save_parameters()
                self.M.result_owner = self.result_owner[best_idx]
                self.M.result_id = best_idx + 1
                self.M.result_pulled = False
        self.M.result_pulled = cluster.broadcast(self.M.result_pulled, object_name="res_pull_status")
        if self.M.visual_callback is not None and not self.M.result_pulled:
            self.M.visual_callback()

    def update_particle_positions(self):
        if cluster.global_rank == 0:
            r1 = np.random.random(self.options.n)[:,None]
            r2 = np.random.random(self.options.n)[:,None]
            self.v *= self.w
            self.v += self.c1*r1*(self.pbest_p - self.p)
            self.v += self.c2*r2*(self.M.p - self.p)
            self.p += self.v

    def enforce_bounds_on_samples(self):
        if cluster.global_rank == 0:
            if self.options.limit_space:
                np.clip(self.p, self.M.p_lb, self.M.p_ub, out=self.p)

    def update_iteration_counter(self):
        if cluster.global_rank == 0:
            self.iter += 1

    def adapt_swarm_weights(self):
        pass

    def update_termination(self):
        if cluster.global_rank == 0:
            self.terminate = any([self.iter >= self.options.max_iter,
                                  self.M.y <= self.options.max_error])

        # broadcast to all workers
        self.terminate = cluster.broadcast(self.terminate, object_name="termination")

    def next_epoch(self):
        epoch_timer = cluster.timer()
        self.extract_results(cluster.map(self.M.loss, self.p))
        self.update_pbest()
        self.update_gbest()
        self.update_particle_positions()
        self.enforce_bounds_on_samples()
        self.adapt_swarm_weights()
        self.update_iteration_counter()
        self.update_termination()
        self.epoch_time = epoch_timer.get_elapsed()
        self.total_time += self.epoch_time
        if self.options.active_callback:
            self.callback(self)

    def train(self, max_iter=100):
        if cluster.global_rank == 0:
            if max_iter is not None:
                self.options.max_iter += max_iter
            else:
                self.options.max_iter += 100
        self.update_termination()

        while not self.terminate:
            self.next_epoch()

def callback(S):
    if cluster.global_rank == 0:
        text = f"""
+===========================================+
Algorithm: {S.name}
Model    : {S.M.model}
+===========================================+
|W = {S.w:8.6f} |C1 = {S.c1:8.6f} |C2 = {S.c2:8.6f} |
+----------------------+----------------+---+
|Epoch number          |{S.iter:15d} | - |
+----------------------+----------------+---+
|Model loss            |{S.M.y:15.6f} | - |
+----------------------+----------------+---+
|Epoch time            |{S.epoch_time:15.4f} | s |
+----------------------+----------------+---+
|Elapsed time          |{S.total_time:15.4f} | s |
+----------------------+----------------+---+
|Mean simulation time  |{S.mean_sim_cpu_time:15.4f} | s |
+----------------------+----------------+---+
|Succesfull            |{S.survived:15d} | - |
+----------------------+----------------+---+
|Failed                |{S.failed:15d} | - |
+----------------------+----------------+---+
|Improved              |{S.improved:15d} | - |
+======================+================+===+
"""
        print(text, end = "\n" if S.terminate else "\033[F"*text.count("\n"))
