import lofi

API = lofi.APIs.open_modelica
model = API('LotkaVoltera.mo', 'LotkaVoltera.ODE')
opt = lofi.optimizers.GRAPSO(model)
for i in range(100):
    opt.step()
lofi.imode(globals())

