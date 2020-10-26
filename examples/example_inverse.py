import lofi

API = lofi.APIs.open_modelica
model = API('HeatTransfer.mo', 'HeatTransfer.InverseProblem')

model.visual_callback.skip = 10 

opt = lofi.optimizers.OPLES(model)
opt.options.max_sigma = 200
opt.options.min_sigma = 0.1
opt.sigma = 200
opt.train(800)

lofi.cluster.imode(globals())
