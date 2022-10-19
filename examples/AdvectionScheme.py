import lofi

API = lofi.APIs.open_modelica
model = API('AdvectionScheme.mo','AdvectionScheme')

opt = lofi.optimizers.GRAPSO(model)
for i in range(200):
    opt.step()

lofi.imode(globals())
