import lofi

API = lofi.APIs.open_modelica
model = API('LotkaVoltera.mo', 'LotkaVoltera.NeuralODE')
opt = lofi.optimizers.OPLES(model)
lofi.imode(globals())

