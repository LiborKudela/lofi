import lofi
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time

# objective function
def valley(x, prms):
    if x is None:
        x = [0.0, 0.0]
    time.sleep(0.01) #fake computational cost
    return (prms[0]-x[0])**2 + (prms[1]-x[1])**2 

api = lofi.APIs.py_function
model = api(valley,
            p_start = [34.7, -53.8], # initial guess (random)
            p_lb=[-100.0,-100.0],    # lower bounds
            p_ub=[100.0, 100.0])     # upper bounds 

#choosing the optimizer
opt = lofi.optimizers.GRAPSO(model)

# Interractive Visualizer
V = lofi.visualizers.Visualizer()

# recepies for Visualizer
@V.figure_recipe("Loss")
def figure():
    fig = px.line(model.log, y="Loss")
    fig.update_layout(uirevision="No change")
    return fig

x_points = np.linspace(-10, 10, 100)
y_points = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_points, y_points)
z_values = (X+1.0)**2 + (Y-1.2)**2
static_valley = go.Contour(
        z=z_values,
        x=x_points, # horizontal axis
        y=y_points, # vertical axis
        contours=dict(
            start=0,
            end=50,
            size=0.2,
        ),
        contours_coloring='lines'
    )


@V.figure_recipe("Valley")
def figure():
    point = go.Scatter(x=[model.p[0]], y=[model.p[1]], marker=dict(size=20))
    return go.Figure(data=[static_valley, point])

# build web app and start the app
V.build_app()
V.start_app()

# inputs for valley (list of arrays)
x = [
    [-1.0, 1.2],
    ]

# optimize problem
for i in range(200):
    opt.step(x=x)

#prin result parameters
model.print_parameters() # should give values in x

# enter interactive mode
lofi.imode(globals())

