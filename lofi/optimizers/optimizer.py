import numpy as np
from ..cluster import cluster
import time

class Options():
    def __init__(self):
        self.n = 40                  # population size
        self.max_iter = 0            # maximum number of iterations
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
        self.epoch_time = 0.0   # time elapsed during last epoch
        self.total_time = 0.0   # total elapsed time during training
        self.iter = 0
        self.options.max_iter = 0
        self.initialize_state()
    
    @cluster.on_master
    def initialize_state(self):
        """This function initializes variables for specific algorithm"""
        pass

    @cluster.on_master
    def extract_results(self, r):
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

    @cluster.on_master
    def pre_gbest_update_actions(self):
        """This function is called prior to gbest_update"""
        pass

    @cluster.on_master
    def update_gbest(self):
        best_idx = np.nanargmin(self.y)
        if self.M.y > self.y[best_idx]:
            self.M.y = self.y[best_idx]
            self.M.p[:] = self.p[best_idx,:]
            self.M.save_parameters()
            self.M.result_owner = self.result_owner[best_idx]
            self.M.result_id = best_idx + 1  # (rank=0) does not simulate 
            self.M.result_pulled = False
    
    @cluster.on_master
    def enforce_bounds_on_samples(self):
        if self.options.limit_space:
            np.clip(self.p, self.M.p_lb, self.M.p_ub, out=self.p)

    @cluster.on_master
    def update_iteration_counter(self):
        self.iter += 1
    
    def broadcast_result_availability(self):
        self.M.result_pulled = cluster.broadcast(self.M.result_pulled)
        if self.M.visual_callback is not None and not self.M.result_pulled:
            self.M.visual_callback()

    @cluster.on_master
    def check_termination(self):
        self.terminate = any([
            self.iter >= self.options.max_iter,
            self.M.y <= self.options.max_error])

    def update_termination(self):
        self.check_termination()
        self.terminate = cluster.broadcast(self.terminate)
    
   @cluster.on_master
   def generate_new_epoch_data(self):
        """This method that assigns new self.p etc."""
        pass 

    @cluster.on_master
    def adaptation(self):
        """This method performs an adaptation procedure"""
        pass

    @cluster.on_master
    def update_console_table(self):
       table = f"""       
+===========================================+
Algorithm: {self.name}
Model    : {self.M.model}
Loss     : {self.M.y:.6f}
Epoch    : {self.iter:d}
+===========================================+
""" 
       print(table, end = "\n" if self.terminate else "\033[F"*table.count("\n"))

    def next_epoch(self):
        epoch_timer = cluster.timer()
        results = cluster.map(self.M.loss, self.p)
        self.extract_results(results)
        self.pre_gbest_update_actions()
        self.update_gbest()
        self.adaptation()
        self.broadcast_result_availability()
        self.generate_new_epoch_data()
        self.enforce_bounds_on_samples()
        self.update_iteration_counter()
        self.update_termination()
        self.epoch_time = epoch_timer.get_elapsed()
        self.total_time += self.epoch_time
        self.update_console_table()
    
    @cluster.on_master
    def add_to_max_iter(self, number):
        self.options.max_iter += number

    def train(self, epochs=100):
        self.add_to_max_iter(epochs)
        self.update_termination()
        while not self.terminate:
            self.next_epoch()
