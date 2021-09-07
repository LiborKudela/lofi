from ..cluster import cluster
import numpy as np
import pandas as pd

class Data():
    def __init__(self, log, state):
        self.log = log
        self.state = state

class Model_api():

    def __init__(self):

        # this parameters information must be specified before super call
        #self.y_names = None
        #self.p_names = None 
        #self.p_lb = None
        #self.p_ub = None
        #self.p_start = None

        # get vector lengths
        self.y_len = len(self.y_names)
        self.m = len(self.p_names)

        # evaluation counter
        self.evals = 0

        # model state with res_file reference info
        if cluster.global_rank == 0:
            self.p = self.p_start
            self.y = self.loss(self.p, result_tag_override=0)[0]
            self.result = self.get_tagged_result(0)
        self.result_owner = None
        self.result_id = None
        self.new_best_result = False

        self.visualizer = None

        self.log = pd.DataFrame(columns=['Evaluations','Loss']) 

    def evaluate(self, prms, result_tag):
        """Evaluates model with parameters prms and saves results.
           Results might be droped into a file.
           The tag serves as a identifier thah might be used in the filename."""
        pass

    def get_tagged_result(self, result_tag):
        """Returns object contaning all results of forward pass
           (might be a content of a file)"""
        pass

    def extract_raw_loss(self, result):
        """Extract all variables coresponding to loss value as a vector from a
        result returned by get_tagged_result function"""
        pass

    def export_result(self, result):
        """Exports result into hdf DataFrame"""
        pass

    @cluster.on_master
    def save_parameters(self):
        pass

    # this section is common to all APIs

    def inf_loss(self, prms, timer):
        return np.inf, np.sum(np.abs(prms)), 1, timer.get_elapsed(), cluster.global_rank

    def real_loss(self, y, prms, timer):
        return np.sum(y), np.sum(np.abs(prms)), 0, timer.get_elapsed(), cluster.global_rank

    def loss(self, prms, result_tag_override=None):

        timer = cluster.timer()
        self.evals += 1

        # resolve name (tag/id) of the result file
        if result_tag_override is not None:
            result_tag = result_tag_override
        else:
            result_tag = cluster.status.Get_tag()

        retcode = self.evaluate(prms, result_tag)

        if retcode != 0:
            return self.inf_loss(prms, timer)
        else:
            result = self.get_tagged_result(result_tag)
            y = self.extract_raw_loss(result)  # y is a np.array of floats

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return self.inf_loss(prms, timer)
        else:
            return self.real_loss(y, prms, timer)

    @cluster.on_master
    def update_state(self, p, y, owner=None, res_id=None):
        "This function updates the current best state of the model"
        self.y = y
        self.p[:] = p
        self.save_parameters()
        self.result_owner = owner
        self.result_id = res_id
        self.new_best_result = True

    def get_total_evals(self):
        """Returns the number of calls to loss function performed on all nodes"""
        return cluster.sum_all(self.evals)

    @cluster.on_master
    def update_visualizer(self):
        if self.visualizer is not None:
            data = Data(self.log, None)
            self.visualizer.update_data(data)

    def update_log(self):
        self.total_evals = self.get_total_evals()
        if cluster.global_rank == 0:
            self.log.loc[len(self.log)]=[
                self.total_evals,
                self.y,
            ]

            self.update_visualizer()

    def pull_result(self):
        """Updates best known result data on master node by pulling from cluster"""

        # Tell every node whether new result is available
        self.new_best_result = cluster.broadcast(self.new_best_result)

        # If newer resuls is available, pull it to master node
        if self.new_best_result:

            # Tell every node what is the best result_id and who owns that file
            data = (self.result_id, self.result_owner)
            self.result_id, self.result_owner = cluster.broadcast(data)

            # If a node is the owner of the wanted file, it reads the data and
            # sends the data to the master node (the zero node)
            if self.result_owner == cluster.global_rank:
                result = self.get_tagged_result(self.result_id)
                cluster.comm.send(result, 0)

            # Master node receives data from the owner of new best results
            if cluster.global_rank == 0:
                self.result = cluster.comm.recv(None, source=self.result_owner)

            # Every node gets notified that newest results have been
            # successfully delivered to the master node
            self.new_best_result = False

    def get_result(self):
        """Returns best of results found so far"""

        # make sure that master has the newest best result
        self.pull_result()

        if cluster.global_rank == 0:
            return self.result

