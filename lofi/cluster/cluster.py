from mpi4py import MPI
import threading
import sys
import time
from collections import Mapping, Container
import numpy as np
import itertools

comm = MPI.COMM_WORLD
size = comm.Get_size()
status = MPI.Status()
global_rank = comm.Get_rank()
name = MPI.Get_processor_name()
namelist = comm.allgather(name)
nodelist = list(set(namelist))
local_rank = len([i for i in namelist[:global_rank] if i == name])

class Options():
    def __init__(self):
        self.reports_active = False
        self.interactive = False

    def activate_reports(self):
        """Activate progress reports of cluster execution."""
        self.reports_active = True

    def deactivate_reports(self):
        """Deactivate progress reports of cluster execution."""
        self.reports_active = False

    def restore_default(self):
        """Reset all options to default state."""
        self.__init__()

    def hardware_info(self):
        if global_rank == 0:
            print("*** Cluster initialization information ***")
            print(f"Node list: {nodelist}")
            print(f"Main Master: GR = {global_rank:3d}, N = {name}, LR = {local_rank:3d}")
        comm.barrier()
        if global_rank != 0:
            if local_rank == 0:
                print(f"Node Master: GR = {global_rank:3}, N = {name} LR = {local_rank:3d}")
            else:
                print(f"Worker: GR = {global_rank:3}, N = {name}, LR = {local_rank:3d}")

    def activate_interactive_mode(self):
        self.interactive = True

options = Options()

class Timer():
    def __init__(self):
        self.start_time = time.time()
        self.last_lap = self.start_time

    def get_elapsed(self):
        self.last_trigered = time.time()
        return self.last_trigered-self.start_time

    def get_lap(self):
        t = time.time()
        lap = t - self.last_lap
        self.last_lap = t
        return time.time() - self.last_lap

def on_master(f):
    def decorated_f(*args, **kwargs):
        if global_rank == 0:
            return f(*args, **kwargs)
    return decorated_f

def on_machine(f):
    def decorated_f(*args, **kwargs):
        if local_rank == 0:
            return f(*args, **kwargs)
    return decorated_f

class Sequential_queue():
    def __init__(self, args, kwargs):

        self.args = args
        self.kwargs = kwargs
        self.set_data_count()
        self.index = 0
        self.received = 0
        self.results = [None]*self.data_count
        self.elapsed = [None]*self.data_count

    def set_data_count(self):
        for arg in self.args:
            if not hasattr(arg, '__iter__'):
                continue
            else:
                self.data_count = len(arg)
                break
        if not hasattr(self, 'data_count'):
            for key, value in self.kwargs.items():
                if not hasattr(value, '__iter__'):
                    continue
                else:
                    self.data_count = len(value)
                    break

    def get(self, arg, index):
        if hasattr(arg, '__iter__'):
            return arg[index]
        else:
            return arg

    def get_args(self, index):
        return tuple((self.get(arg, index) for arg in self.args))

    def get_kwargs(self, index):
        return {key: self.get(value, index) for key, value in self.kwargs.items()}

    def next_items(self, n=None):
        start = self.index
        if n is None:
            end = self.data_count
        else:
            end = min(self.index + n, self.data_count)

        for i in range(start, end):
            yield self.get_args(self.index), self.get_kwargs(self.index), self.index
            self.index += 1

    def empty_results(self):
        while self.received != self.data_count:
            yield True

    def __len__(self):
        return self.data_count

    def put_result(self, result_data):
        result, elapsed, index = result_data
        self.results[index] = result
        self.elapsed[index] = elapsed
        self.received += 1

#TODO: change for nd not just 2d
class _2D_Product_queue():
    def __init__(self, args):
        self.args = args
        self.shape = (len(args[0]), len(args[1]))
        self.data_count = self.shape[0]*self.shape[1]
        self.flat_index = 0
        self.received = 0
        self.results = [[None]*self.shape[1] for i in range(self.shape[0])]
        self.elapsed = [[None]*self.shape[1] for i in range(self.shape[0])]

    def flat_index_to_2D_index(self, flat_index):
        index_0 = int(flat_index/self.shape[1])
        index_1 = flat_index % self.shape[1]
        return (index_0, index_1)

    def get_current_index(self):
        return self.current_flat_index

    def next_items(self, n=None):
        start = self.flat_index
        if n is None:
            end = self.data_count
        else:
            end = min(self.flat_index + n, self.data_count)

        for i in range(start, end):
            index = self.flat_index_to_2D_index(self.flat_index)
            yield (self.args[0][index[0]], self.args[1][index[1]]), {}, index
            self.flat_index += 1

    def empty_results(self):
        while self.received != self.data_count:
            yield True

    def __len__(self):
        return self.data_count

    def put_result(self, result_data):
        result, elapsed, index = result_data
        self.results[index[0]][index[1]] = result
        self.elapsed[index[0]][index[1]] = elapsed
        self.received += 1

class Worker_killer():
    def __init__(self):
        pass
worker_killer = Worker_killer()

def master(queue):
    """
    This function dynamically saturates workers with data.
    It must be executed on master (global_rank=0) only.
    After all data have been used the function will colect last results
    and send tag=0 to workes which breaks their worker loop.
    The function called worker must therefore be executed on
    all workes while this functin is being employed.
    """
    if size == 1:
        print("Run with more cores -> 1 for master rest for the workers")
        return None

    # saturating workers
    for i, data in enumerate(queue.next_items(n=size-1), 1):
        comm.send(data, dest=i)

    # receiving results and keeping workers saturated (dynamic scheduling)
    for data in queue.next_items():
        result_data = comm.recv(None, source=MPI.ANY_SOURCE, status=status)
        queue.put_result(result_data)
        comm.send(data, dest=status.Get_source())

    # collect last results
    for i in queue.empty_results():
        result_data = comm.recv(None, source=MPI.ANY_SOURCE, status=status)
        queue.put_result(result_data)

    # sending termination signal to the workers
    for i in range(1, size):
        comm.send(worker_killer, dest=i) # tag=0 breaks worker loop

    return queue.results, queue.elapsed

def worker(work):
    """This function waits for data from master, executes work on them,
    send result to master and waits for another instruction from master.
    Worker_killer instance breaks out of this loop.
    """
    while True:
        data = comm.recv(None, source=0)
        if isinstance(data, Worker_killer):
            break
        timer = Timer()
        args, kwargs, index = data
        result = work(*args, **kwargs)
        elapsed = timer.get_elapsed()
        comm.send((result, elapsed, index), dest=0)

def sequential_map(work, *args, **kwargs):
    """Order preserving scheduling of work with data as argument"""
    if global_rank == 0:
        queue = Sequential_queue(args, kwargs)
        return master(queue)
    else:
        worker(work)
        return None, None

def _2d_product_map(work, *args):
    """Order preserving scheduling of work with data as argument"""
    if global_rank == 0:
        queue = _2D_Product_queue(args)
        return master(queue)
    else:
        worker(work)
        return None, None

def broadcast(object, root=0):
    """Broadcast by assignment."""
    return comm.bcast(object, root=root)

def Broadcast(object, root=0):
    """Broadcast to buffer. e.g. numpy arrays."""
    comm.Bcast(object, root=root)

def collect(data, source, dest=0):
    """Ask for data from source and deliver to the destination (defaults to master)"""
    if global_rank == source:
        comm.send(data, dest=dest, tag=0)
        return data
    elif global_rank == dest:
        return comm.recv(None, source=source, tag=0)

def sum_all(data, dest=0):
    "Returns sum acumulated across all nodes"
    return comm.reduce(data, op=MPI.SUM, root=dest)

def deep_getsizeof(o, ids):
    if id(o) in ids:
        return 0
    r = sys.getsizeof(o)
    ids.add(id(o))
    if isinstance(o, str) or isinstance(o, bytes):
        return r
    if isinstance(o, Mapping):
        return r + sum(deep_getsizeof(k, ids) + deep_getsizeof(v, ids) for k, v in o.iteritems())
    if isinstance(o, Container):
        return r + sum(deep_getsizeof(x, ids) for x in o)
    return r
