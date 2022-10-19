import lofi

def valley(x, prms):
    if x is None:
        x = [0.0, 0.0]
    return (prms[0]-x[0])**2 + (prms[1]-x[1])**2 

api = lofi.APIs.py_function
model = api(valley,
            p_start = [34.7, -53.8],
            p_lb=[-100.0,-100.0],
            p_ub=[100.0, 100.0])

opt = lofi.optimizers.GRAPSO(model)

# inputs for valley (lst of lists)
x = [
    [-1.0, 1.2]
    ]
for i in range(200):
    opt.step(x=x)

model.print_parameters() # should give values in x
lofi.imode(globals())

