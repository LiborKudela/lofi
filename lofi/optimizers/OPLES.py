import numpy as np
from ..cluster import cluster
import time
import matplotlib.pyplot as plt

class options():
    def __init__(self):
        self.n = 40                  # populations size
        self.max_sigma = 1e-1        # max sampling deviation
        self.min_sigma = 1e-8        # min sampling deviation
        self.sigma_init = 1e-2       # initial sigma
        self.success_rule = 0.11     # this seems to work
        self.max_iter = 0            # maximum number of iterations
        self.max_error = -np.inf     # maximum allowed error
        self.limit_space = True      # optional coordinate bound control
        self.sparse_program = False  # optional regularization for sparsity
        self.sparsity_weight = 1.0   # weight to penalize density of program
        self.active_callback = True  # activates reports to terminal

class OPLES():
    def __init__(self, M=None):
        self.options = options()
        self.callback = callback
        self.name = "OPLES"
        if M is not None:
            self.connect_model(M)
            self.restart()

    def connect_model(self, M):
        self.M = M  # reference to a model instance

    def disconnect_model(self):
        self.M = None

    def restart(self):
        self.terminate = False  # termination condition state
        self.epoch_time = 0.0   # time elapsed during last epoch
        self.total_time = 0.0   # total elapsed time during training
        self.options.max_iter = 0

        if cluster.global_rank == 0:
            self.sigma = self.options.sigma_init  # initial sampling deviance
            self.success_rate = 0
            self.Ldrop_rate = 0                   # loss decrease rate
            self.C_rate = 0                       # convergence rate
            self.iter = 0                         # iteration counter
            self.velocity = np.zeros(self.M.m)    # parental velocity
            self.velocity_norm = 0                # parental velocity norm
        else:
            self.p = None  # this is necessary for cluster.map call

    def extract_results(self, r):
        if cluster.global_rank == 0:
            # make r into an array so each value type corresponds to a single row
            # r[0,:] losses, r[1,:] densities, r[2,:] return codes, r[3,:] sim times
            r = np.array(r).transpose()

            # read results by type (which is by rows)
            self.y = r[0,:]
            if self.options.sparse_program:
                self.y += r[1,:]
            self.failed = np.count_nonzero(r[2,:])
            self.survived = self.options.n - self.failed
            self.mean_sim_cpu_time = np.sum(r[3,:])/self.options.n
            self.result_owner = r[4,:]

    def update_gbest(self):
        if cluster.global_rank == 0:
            self.idx = np.nanargmin(self.y)
            self.prev_gbest_y = self.M.y
            self.prev_gbest_p = np.copy(self.M.p)

            if self.M.y >= self.y[self.idx]:
                self.M.y = self.y[self.idx]
                self.M.p[:] = self.p[self.idx, :]
                self.M.save_parameters()
                self.M.result_owner = self.result_owner[self.idx]
                self.M.result_id = self.idx + 1
                self.M.result_pulled = False
        self.M.result_pulled = cluster.broadcast(self.M.result_pulled, object_name="res_pull_status")
        if self.M.visual_callback is not None and not self.M.result_pulled:
            self.M.visual_callback()

    def update_sigma(self):
        if cluster.global_rank == 0:
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

    def generate_new_samples(self):
        if cluster.global_rank == 0:
            self.z = np.random.normal(0, 1, (self.options.n, self.M.m))
            self.p = self.M.p + self.sigma*self.z + self.velocity
            if self.velocity_norm > 1e-7:
                self.p[0] = self.M.p + 0.9*self.velocity
                self.p[1] = self.M.p + 1.1*self.velocity

    def enforce_bounds_on_samples(self):
        if cluster.global_rank == 0 and self.options.limit_space:
            np.clip(self.p, self.M.p_lb, self.M.p_ub, out=self.p)

    def update_iteration_counter(self):
        if cluster.global_rank == 0:
            self.iter += 1

    def update_termination(self):
        if cluster.global_rank == 0:
            self.terminate = any([self.iter >= self.options.max_iter,
                                  self.M.y <= self.options.max_error])

        # broadcast to all workers
        self.terminate = cluster.broadcast(self.terminate, object_name="terminate")

    def next_epoch(self):
        epoch_timer = cluster.timer()
        self.generate_new_samples()
        self.enforce_bounds_on_samples()
        self.extract_results(cluster.map(self.M.loss, self.p))
        self.update_gbest()
        self.update_sigma()
        self.update_iteration_counter()
        self.update_termination()
        self.epoch_time = epoch_timer.get_elapsed()
        self.total_time += self.epoch_time
        if self.options.active_callback:
            self.callback(self)

    def train(self, max_iter=None):
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
+======================+================+===+
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
|Sampling deviation    |{S.sigma:15.10f} | - |
+----------------------+----------------+---+
|Parental velocity     |{S.velocity_norm:15.10f} | - |
+----------------------+----------------+---+
|Success rate          |{100*S.success_rate:8.4f}/{100*S.options.success_rule:5.3f} | % |
+----------------------+----------------+---+
|Convergence rate      |{100*S.C_rate:15.4f} | % |
+----------------------+----------------+---+
|Loss drop rate        |{S.Ldrop_rate:15.4f} | - |
+======================+================+===+
"""
        print(text, end = "\n" if S.terminate else "\033[F"*text.count("\n"))
