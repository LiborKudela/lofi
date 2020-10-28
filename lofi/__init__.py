from .cluster import cluster
from . import APIs
from . import optimizers
from .visualizers import visualizers
from time import sleep

# lofi has interactive mode that can be call using lofi.imode(globals())
def imode(globals):
    if cluster.global_rank == 0:
        print("Entering interactive mode...")
    while True:

        # get command on master node console
        cluster.comm.barrier()
        if cluster.global_rank == 0:
            command = input("lofi imode >>>")
        else:
            command = None

        # wait for command in non-blocking way
        req = cluster.comm.Ibarrier()
        while not req.test()[0]:
            sleep(0.01)
        req.wait()
        command = cluster.broadcast(command)

        # execute the command on every node
        try:
            if command == "imode.leave()":
                break
            else:
                exec(command, globals)
        except Exception as err:
            if cluster.global_rank == 0:
                detail = err.args[0]
                print(detail)
