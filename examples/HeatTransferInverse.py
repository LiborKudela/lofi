import lofi

API = lofi.APIs.open_modelica
model = API('HeatTransfer.mo', 'HeatTransfer.InverseProblem')

opt = lofi.optimizers.RMSPropES(model, sigma=1, lr=1)
for i in range(200):
    opt.step()

lofi.imode(globals())
