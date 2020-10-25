import LOFI

API = LOFI.APIs.open_modelica
model = API('SDEWES_examples/LotkaVoltera.mo',
            'LotkaVoltera.NeuralODE', fast_storage='/dev/shm')

opt = LOFI.optimizers.OPL_ES(model)

LOFI.cluster.enter_imode(globals())

