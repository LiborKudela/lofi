from ..cluster import cluster
import  numpy as np
from .model_api import Model_api

class py_function(Model_api):
    def __init__(self, f, p_start, p_lb, p_ub): 
        self.f = f
        self.model = f.__name__

        self.p_start = p_start
        self.p_lb = np.array(p_lb)
        self.p_ub = np.array(p_ub)
        self.y_names = None

        self.m = len(self.p_start)

        super().__init__()

    def forward(self, x=None, prms=None, output=None):
        """Evaluate function with imput x and parameters prms"""
        # add result checks to avoid singularities and such
        return self.f(x, prms)

    @cluster.on_master 
    def print_parameters(self):
        print(self.p)

