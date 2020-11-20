from .optimizer import Optimizer, Options, cluster, np

class Template_options(Options):
    def __init__(self):
        super().__init__()
        self.additional_option = "default_value"

class Template(Optimizer):
    def __init__(self, M):
        super().__init__(M, Template_options)
        self.name = "Template"

    @cluster.on_master
    def initialize_state(self):
        """This function initializes variables for specific algorithm"""
        pass

    @cluster.on_master
    def pre_gbest_update_actions(self):
        """This function is called prior to gbest_update"""
        pass

    @cluster.on_master
    def generate_new_epoch_data(self):
        """This method that assigns new self.p for next epoch"""
        pass 

    @cluster.on_master
    def adaptation(self):
        """This method performs an adaptation procedure after gbest_update"""
        pass
