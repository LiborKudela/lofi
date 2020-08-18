import matplotlib.pyplot as plt
import cluster

class default_visual_callback:
    def __init__(self, API):      
        plt.ion()
        self.API = API
        self.refresh_data = self.initialize_window
        self.refresh_attempt_counter = 0
        self.skip = 50
        self.save = False

    def initialize_window(self):  
        res = self.API.get_result()
        if cluster.global_rank == 0:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel("Time (s)")
            self.line = []
            for name in self.API.plot_vars:
                data = res.getVarArray([name])[0:,:]
                line, = self.ax.plot(data[0,:], data[1,:], label=name)
                self.line.append(line)
            self.ax.legend()
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.1)
        self.refresh_data = self.update_existing_window

    def update_existing_window(self):
        res = self.API.get_result()
        if cluster.global_rank == 0:
            for i, name in enumerate(self.API.plot_vars):
                ydata = res.getVarArray([name],withAbscissa=False)
                self.line[i].set_ydata(ydata)
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def __call__(self):
        if self.refresh_attempt_counter % self.skip == 0:
            self.refresh_data()
            if cluster.global_rank == 0 and self.save:
                self.fig.savefig(f'pictures/pic{self.refresh_attempt_counter}')
        self.refresh_attempt_counter += 1
