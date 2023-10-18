import numpy as np
from lammps import lammps, PyLammps
from dm.main import utils

class LAMMPS():
    def __init__(self,input_dir, save_dir=None,mass=None, init_pos=None, dim=3, temp=1,
        integrator=None, ran_seed=None, cell_len=1,name=''):
        self.lmp = lammps()
        self.temp = temp
        if ran_seed == None:
            self.ran_seed = np.random.randint(0,10000)
        else:
            self.ran_seed = ran_seed
        params={"CELL_LEN" : cell_len}
        if save_dir is None:
            save_dir = input_dir
        self.set_input_params(params,input_dir,save_dir+name+"_input.lmp")
        if integrator == "langevin":
            self.lmp.command("fix f all langevin %f %f 0.1 %d zero yes"%(self.temp,self.temp,self.ran_seed))
        self.dim = dim
        self.pylmp = PyLammps(ptr=self.lmp)
        self.nparticles = self.pylmp.system.natoms
        if init_pos is not None:
           self.set_position(init_pos)
        if mass == None:
            self.mass = np.ones(self.nparticles)
        else:
            self.mass = np.broadcast_to(np.array(mass),self.nparticles)
  
            
    def set_input_params(self, params, template, input):
        with open(template, "r") as template:
            with open (input,"w") as output:
                for line in template:
                    output.write(line.format(**params))
        self.lmp.file(input)


    def command(self, str):
        self.lmp.command(str)

    def get_potential(self,position=None):
        if position is not None:
            self.set_position(position)
        self.pylmp.run(0)
        return self.lmp.numpy.extract_variable("energy")

    def get_position(self):
        return np.array(np.ctypeslib.as_array(self.lmp.gather_atoms("x",1,3)))


    def set_position(self,position):
        if isinstance(position,str):
            position = utils.load_position(position).reshape(-1,self.dim)
        else:
            position = position.reshape(-1,self.dim)
        id = self.lmp.numpy.extract_atom("id")
        for i,index in zip(range(self.nparticles), id):
            self.pylmp.atoms[i].position= position[index-1]

    def set_velocity(self,velocity):
        velocity = velocity.reshape(-1,self.dim)
        id = self.lmp.numpy.extract_atom("id")
        for i,index in zip(range(self.nparticles), id):
            self.pylmp.atoms[i].velocity= velocity[index-1]


    def integration_step(self,path_len=1,dt=None, init_pos=None ):
        if init_pos is not None:
            self.set_position(init_pos)
        if dt is not None:
            self.pylmp.command("timestep %f"%dt)
        self.pylmp.run(int(path_len))
        position = self.get_position()
        potential = self.lmp.numpy.extract_variable("energy")
        return position, potential
