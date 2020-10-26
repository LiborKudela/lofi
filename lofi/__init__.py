from .cluster import cluster
from . import APIs
from . import optimizers
from .visualizers import visualizers

# INTERACTIVITY
prompt = "LOFI >>> "

# custom LOFI way - blocking! but Xforwarding works over ssh -Y
def imode(globals):
    if cluster.global_rank == 0:
        import traceback
        print("Entering interactive mode...")
    while True:
        cluster.comm.barrier()
        if cluster.global_rank == 0:
            request = input(prompt)
        else:
            request = None

        request = cluster.broadcast(request)
        try:
            exec(request, globals)
        except Exception as err:
            if cluster.global_rank == 0:
                description = 'source string'
                error_class = err.__class__.__name__
                detail = err.args[0]
                cl, exc, tb = sys.exc_info()
                print("%s at line %d of %s: %s" % (error_class,
                                                   line_number,
                                                   description,
                                                   detail))

# using build in python interactive mode (python3 -i script.py)
if cluster.global_rank == 0:
    cluster.sys.ps1 = prompt
else:
    cluster.sys.ps1 = ""
