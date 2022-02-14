import numpy as np
from ..cluster import cluster
import time

class Options():
    def __init__(self):
        self.max_error = -np.inf     # maximum allowed error
        self.limit_space = True      # optional coordinate bound control
        self.sparse_program = False  # optional regularization for sparsity
        self.sparsity_weight = 1.0   # weight to penalize density of program
        self.active_callback = True  # activates reports to terminal

class Optimizer():
    def __init__(self, M=None, options=Options):
        self.p = None  # this is necessary for cluster.map call
        self.options = options()
        if M is not None:
            self.connect_model(M)
            self.restart()

    def connect_model(self, M):
        self.M = M  # reference to model instance

    def disconnect_model(self):
        self.M = None

    def restart(self):
        self.terminate = False  # termination condition state
        self.step_time = 0.0   # time elapsed during last epoch
        self.total_time = 0.0   # total elapsed time during training
        self.iter = 0
        self.options.max_iter = 0
        self.initialize_state()
        self.initialize_parameter_array()
        self.results = []

    @cluster.on_master
    def initialize_state(self):
        """This function initializes variables for specific algorithm"""
        pass

    def initialize_parameter_array(self):
        if cluster.global_rank == 0:
            shape = self.p.shape
            self.p_array = np.zeros((shape[0] + 1, shape[1]))
        else:
            self.p_array = None

    @cluster.on_master
    def fill_parameter_array(self):
        self.p_array[:-1] = self.p  # add new population of the method
        self.p_array[-1] = self.M.p # add current gbest (loss can be dynamic)

    def evaluate_samples(self, x=None):
        if x is None:
            data = cluster.sequential_map(self.M.eval_loss, self.p_array)
        else:
            data = cluster._2d_product_map(self.M.eval_loss, self.p_array, x)
        self.results, _ = cluster.sequential_map(self.M.loss, data[0])

        if cluster.global_rank == 0:
            self.mean_sim_cpu_time = np.mean(np.array(data[1]))
            self.M.y = self.results[-1]
            self.y[:] = np.array(self.results[:-1])

            # count succesfully completed evaluations
            total = len(self.y)
            self.failed = np.count_nonzero(self.y == np.inf)
            self.survived = total - self.failed

    @cluster.on_master
    def resolve_map_elapsed(self,):
        self.mean_sim_cpu_time = np.mean(np.array(self.elapsed))

    @cluster.on_master
    def update_model(self):
        """This method calculates and assigns newest model parameters"""
        pass

    def update_log(self):
        self.M.update_log()

    @cluster.on_master
    def enforce_bounds_on_samples(self):
        if self.options.limit_space:
            np.clip(self.p, self.M.p_lb, self.M.p_ub, out=self.p)

    @cluster.on_master
    def update_iteration_counter(self):
        self.iter += 1

    @cluster.on_master
    def generate_new_epoch_data(self):
        """This method that assigns new self.p to be evaluated."""
        pass

    @cluster.on_master
    def update_console_table(self):
       tbl = f"""
\033[K+===========================================+
\033[KAlgorithm : {self.__class__.__name__}
\033[KModel     : {self.M.model}
\033[KLoss      : {self.M.y:.6f}
\033[KEpoch     : {self.iter:d}
\033[KEvals     : {self.M.total_evals:d}
\033[KStep Time : {self.step_time:.4f}
\033[KOpt Time  : {self.total_time:.4f}
\033[KAvg time  : {self.mean_sim_cpu_time:.4f}
\033[KSuccess   : {self.survived:d}
\033[KFailure   : {self.failed:d}
\033[K+===========================================+
"""
       print(tbl)

    def step(self, x=None, n=1):
        for i in range(n):
            step_timer = cluster.Timer()
            self.generate_new_samples()
            self.enforce_bounds_on_samples()
            self.fill_parameter_array()
            self.evaluate_samples(x)
            self.update_model()

            self.update_iteration_counter()
            self.update_log()
            self.update_console_table()
            self.step_time = step_timer.get_elapsed()
            self.total_time += self.step_time
