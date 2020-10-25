import LOFI

API = LOFI.APIs.open_modelica
model = API('SDEWES_examples/LotkaVoltera.mo',
            'LotkaVoltera.ODE', fast_storage='/dev/shm')

model.visual_callback.skip = 5
model.visual_callback.save = True
opt = LOFI.optimizers.GRAPSO(model)
opt.train(200)

LOFI.cluster.enter_imode(globals())

