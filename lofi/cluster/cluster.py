from mpi4py import MPI
import sys
import time
from collections import Mapping, Container
import numpy as np

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

class timer():
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

class queue():
    def __init__(self, data):
        self.data_items = data
        self.start_len = len(data)
        self.next_item_idx = 0

    def preallocate_results(self):
        return [None]*self.start_len

    def get_next_item(self):
        if self.next_item_idx == self.start_len:
            return None
        item = self.data_items[self.next_item_idx]
        self.next_item_idx += 1
        return item

class Reporter():
    """Progress of execution as a progress bar"""
    def __init__(self, header, init_progress=0.0):
        self.timer = timer()
        self.progress = init_progress
        self.bar_fill = int(self.progress)

        if options.reports_active:
            print(header)
            self.update = self.active_call
        else:
            self.update = self.inactive_call

    def draw_bar(self):
        print("|" + "\u25AE" * self.bar_fill + "-" * (100-self.bar_fill) + "|", end="")

    def print_progress(self):
        print(": %.1f %%, : %.3f s" % (self.progress, self.timer.get_elapsed()), end="")

    def active_call(self, progress):
        self.progress = progress
        self.bar_fill = int(progress)

        sys.stdout.write(u"\u001b[1000D")
        self.draw_bar()
        self.print_progress()
        sys.stdout.flush()

        if self.bar_fill == 100:
            print()

    def inactive_call(self, progress):
        pass

def master(data=None, msg="msg-empty"):
    """
    This function dynamically saturates workers with data.
    It must be executed on master (global_rank=0) only.
    After all data have been used the function will colect last results
    and send tag=0 to workes which breaks their worker loop.
    The function called worker must therefore be executed on
    all workes while this functin is being employed.
    """
    reporter = Reporter(msg)
    queued_data = queue(data)
    all_results = queued_data.preallocate_results()

    if size == 1:
        print("Run with more cores -> 1 for master rest for the workers")
        return None

    # saturating workers (initial spawn)
    tag = 0
    for i in range(1, size):
        next = queued_data.get_next_item()
        if next is None:
            break
        tag += 1
        comm.send(next, dest=i, tag=tag)

    # receiving results and keeping workers saturated (dynamic scheduling)
    finished = 0
    while 1:
        next = queued_data.get_next_item()
        if next is None:
            break
        result = comm.recv(None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        all_results[status.Get_tag() - 1] = result
        finished += 1
        reporter.update(100*finished/queued_data.start_len)
        tag += 1
        comm.send(next, dest=status.Get_source(), tag=tag)

    # receiving last results
    for i in range(1,min(size, queued_data.start_len+1)):
        result = comm.recv(None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        all_results[status.Get_tag() - 1] = result
        finished += 1
        reporter.update(100*finished/queued_data.start_len)

    # sending termination signal of the worker loops
    for i in range(1,size):
        comm.send(None, dest=i, tag=0) # tag=0 kills breaks worker loop
    return all_results

def worker(work=None):
    """This function waits for data from master, executes work on them,
    send result to master and waits for another instruction from master.
    tag = 0 breaks out of this loop.
    """
    while 1:
        data = comm.recv(None, source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if not tag:
            break
        comm.send(work(data), dest=0, tag=tag)

def map(work, data):
    """Order preserving scheduling of work with data as argument
       (basically order preserving parallel for loop)"""
    if global_rank == 0:
        return master(data, msg="Dynamic mapping: " + work.__name__)
    else:
        worker(work)

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
