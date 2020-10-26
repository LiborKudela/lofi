import lofi

API = lofi.APIs.open_modelica
model = API('AdvectionScheme.mo','AdvectionScheme')

model.visual_callback.skip = 25
opt = lofi.optimizers.GRAPSO(model)
opt.train(200)

lofi.cluster.imode(globals())
