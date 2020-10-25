import LOFI

API = LOFI.APIs.open_modelica
model = API('SDEWES_examples/HeatTransfer.mo',
            'HeatTransfer.InverseProblem',
             fast_storage='/dev/shm')

model.visual_callback.skip = 10 

opt = LOFI.optimizers.OPL_ES(model)
opt.options.max_sigma = 200
opt.options.min_sigma = 0.1
opt.sigma = 200
opt.train(800)

LOFI.cluster.enter_imode(globals())
