import lofi
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

data_time = 25.0
data_size = 1000
step_size = data_time/data_size
batch_size = 20
batch_time = 10

model = lofi.APIs.open_modelica('Cube_model.mo',
                                'Cube_model.Training',
                                solver='ida')

data = model.forward({
        "stopTime": data_time,
        "stepSize": step_size,
        'u0[1]': 2,
        'u0[2]': 0})

true_u = data.getVarArray(["data.u[1]", "data.u[2]"], withAbscissa=False)


def get_batch():
    s = np.random.choice(np.arange(data_size - batch_time), batch_size, replace=False)
    inputs = []
    for i in s:
        d = {"stopTime": batch_time*step_size,
             "stepSize": step_size,
             'u0[1]': true_u[0, i],
             'u0[2]': true_u[1, i]}
        inputs.append(d)
    return inputs

opt = lofi.optimizers.RMSPropES(model, lr=2e-3, sigma=1e-6, n=19, alpha=0.99)

V = lofi.visualizers.Visualizer()

@V.display_recipe()
def displays():
    res = model.forward({
        "stopTime": data_time,
        "stepSize": step_size,
        'u0[1]': 2,
        'u0[2]': 0},output=model.y_names)
    d = dict(loss=np.sum(res))
    return d

@V.figure_recipe("Loss")
def figure():

    fig = px.line(model.log, y="Loss")
    fig.update_layout(uirevision="No change")
    return fig

@V.figure_recipe("Model state")
def figure2():
    res = model.forward({
        "stopTime": data_time,
        "stepSize": step_size,
        'u0[1]': 2,
        'u0[2]': 0})
    fig = make_subplots(rows=1, cols=2)

    #first figure
    row, col = 1, 1
    fig.add_trace(go.Scatter(x=res['neuralODE.u[1]'], y=res['neuralODE.u[2]']),row=row,col=col)
    fig.add_trace(go.Scatter(x=res['data.u[1]'], y=res['data.u[2]']),row=row,col=col)

    #second figure
    row, col = 1, 2
    fig.add_trace(go.Scatter(y=res['neuralODE.u[1]'],line={"color":"red"}),row=row,col=col)
    fig.add_trace(go.Scatter(y=res['data.u[1]'],line={"color":"red","dash":"dash"}),row=row,col=col)
    fig.add_trace(go.Scatter(y=res['neuralODE.u[2]'],line={"color":"green"}),row=row,col=col)
    fig.add_trace(go.Scatter(y=res['data.u[2]'],line={"color":"green","dash":"dash"}),row=row,col=col)

    fig.update_layout(
        {
            "yaxis":{"range":[-2,2]},
            "xaxis":{"range":[-2,2]},
            "yaxis2":{"range":[-2,2]},
            "xaxis2":{"range":[0,1000]}
        },
        uirevision = "NO change",
    )
    return fig

V.build_app()
V.start_app()

init_methods = lofi.initializers
model.init("weight", init_methods.normal, mean=0.0, std=0.1)
model.init("bias", init_methods.constant, value=0.0)

def epoch(n):
    for i in range(n):
        x = get_batch()
        opt.step(x=x)

epoch(2000)

lofi.imode(globals())
