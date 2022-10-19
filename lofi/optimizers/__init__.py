# standard PSO with constant momentum weight 
from .PSO import PSO

# Greedy random adaptation PSO 
from .GRAPSO import GRAPSO

# Basic evolution strategy estimated gradient descend (antithetic samples)
from .VanillaES import VanillaES

# RMSProp but the gradient is estimated by Vanilla ES
from .RMSPropES import RMSPropES

# Vanilla random search steping to best sample
from .RS import RS
