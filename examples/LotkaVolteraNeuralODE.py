import lofi

API = lofi.APIs.open_modelica
model = API('LotkaVoltera.mo', 'LotkaVoltera.NeuralODE')
opt = lofi.optimizers.RMSPropES(model)
for i in range(100):
    opt.step()
lofi.imode(globals())

