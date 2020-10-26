import lofi

API = lofi.APIs.open_modelica
model = API('SDEWES_examples/LotkaVoltera.mo', 'LotkaVoltera.ODE')

model.visual_callback.skip = 5
model.visual_callback.save = True
opt = lofi.optimizers.GRAPSO(model)
opt.train(200)

lofi.cluster.imode(globals())

