import LOFI

API = LOFI.APIs.open_modelica
model = API('SDEWES_examples/AdvectionScheme.mo',
            'AdvectionScheme', fast_storage='/dev/shm')

model.visual_callback.skip = 25
opt = LOFI.optimizers.GRAPSO(model)
opt.train(200)

LOFI.cluster.enter_imode(globals())
