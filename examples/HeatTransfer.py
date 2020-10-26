import lofi

API = lofi.APIs.open_modelica
model = API('LotkaVoltera.mo','LotkaVoltera.ODE')

opt = lofi.optimizers.GRAPSO(model)
opt.train(1)
lofi.imode(globals())

