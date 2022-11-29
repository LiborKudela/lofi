
<p align="center">
  <img src="https://github.com/LiborKudela/lofi/blob/master/logo.svg?raw=True" width="300"/>
</p>

**Lofi is a simple derivative free, distributed (MPI) optimization toolset for black-box (or non-differentiable) models.**

**Instalation instruction using pip (Lixux)**  
python3 -m pip install git+https://github.com/LiborKudela/lofi.git


**How to run examples:**  
Start up terminal and Navigate to /examples directory.  
Then run command: ```mpirun -n 4 python3 py_function_example.py```

**Available Interfaces:**
* OpenModelica
* Python functions

**Optimizers:**
* PSO (Particle swarm optimizer)
* GRAPSO (Greedy random adaptation particle swarm optimizer)
* RMSPropES (Gradientless version of RMSProp gradient descent)
* RS (Random search)
* VanilaES (Gradientless descend where grad is aproximated very roughly)
* More algorithms maybe in the future
