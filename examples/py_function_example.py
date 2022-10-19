import lofi

# objective function
def valley(x, prms):
    if x is None:
        x = [0.0, 0.0]
    return (prms[0]-x[0])**2 + (prms[1]-x[1])**2 

api = lofi.APIs.py_function
model = api(valley,
            p_start = [34.7, -53.8], # initial guess (random)
            p_lb=[-100.0,-100.0],    # lower bounds
            p_ub=[100.0, 100.0])     # upper bounds 

opt = lofi.optimizers.GRAPSO(model)

# inputs for valley (list of arrays)
x = [
    [-1.0, 1.2],
    [-1.0, 1.2]
    ]
for i in range(200):
    opt.step(x=x)

model.print_parameters() # should give values in x
lofi.imode(globals())

