from ..cluster import cluster
import numpy as np
import pandas as pd

class Model_api():

    def __init__(self):

        # this parameters information must be specified before super call
        #self.y_names = None
        #self.p_names = None 
        #self.p_lb = None
        #self.p_ub = None
        #self.p_start = None

        # evaluation counter
        self.evals = 0

        # model state with res_file reference info
        self.p = np.array(self.p_start)
        self.y = self.eval_loss(self.p)

        self.log = pd.DataFrame(columns=['Evaluations','Loss']) 

    def forward(self, x=None, prms=None, output=None):
        """Evaluates model with parameters prms and saves results.
           Results might be droped into a file."""
        pass

    @cluster.on_master
    def save_parameters(self):
        """Gets called when everytime model improves"""
        pass

    # this section is common to all APIs

    def loss(self, y):
        return np.sum(y)

    def eval_loss(self, prms, x=None):

        self.evals += 1
        y = self.forward(x=x, prms=prms, output=self.y_names)

        if y is None:
            return np.inf
        else:
            return y

    def update_state(self, p):
        "This function updates the current best state of the model"
        if cluster.global_rank == 0:
            self.p[:] = p
        #self.save_parameters()
        data = (self.p, self.y)
        self.p, self.y = cluster.broadcast(data)

    def get_total_evals(self):
        """Returns the number of calls to loss function performed on all nodes"""
        return cluster.sum_all(self.evals)

    def update_log(self):
        self.total_evals = self.get_total_evals()
        if cluster.global_rank == 0:
            self.log.loc[len(self.log)]=[
                self.total_evals,
                self.y,
            ]
