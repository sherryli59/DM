import numpy as np
import argparse
import mdtraj as md
import simtk.unit as unit
import openmm.app as app
import openmm as omm
from sys import stdout
from mdtraj.reporters import HDF5Reporter
import torch
import os

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

def _parse_CLAs():
    parser = argparse.ArgumentParser(description="Run a \
    simulation of Lennard Jones liquid")
    parser.add_argument('--replicate', type=int, default=0, help='index of \
    statistical replicate.')
    parser.add_argument('--temp', type=float, default=300, help='Temperature\
    defaults to 300')              
    parser.add_argument('--output', type=str, default=None, help='Where to \
    save the output of the simulation')
    parser.add_argument('--nsteps', type=int, default=1e5, help='How many \
    samples to collect')
    parser.add_argument('--burnin', type=int, default=1e4, help='Number of \
    steps to burn in the trajectory for.')
    parser.add_argument('--platform', type=str, default="CUDA", help='Default platform \
    is CUDA.')
    return parser

class ADP():
    def __init__(self,temp=300, burnin=5000,dt=0.002, save_traj=True,filename="adp",**kwargs):
        self.dir = os.path.realpath(os.path.dirname(__file__))
        self.dt = dt*unit.picosecond
        self.temp = temp*unit.kelvin

        self.save_traj = save_traj
        self.filename = filename
        self.system = self._init_system()
        self.integrator = self._init_langevin_integrator()
        self.simulation = self._init_simulation()
        


    def _init_system(self):
        system = omm.System()
        prmtop = app.AmberPrmtopFile(os.path.join(self.dir, 'input/adp_implicit.prmtop'))
        system = prmtop.createSystem(implicitSolvent=app.OBC1, nonbondedCutoff=None,rigidWater=True)
        CMMotionRemover = omm.CMMotionRemover(1)
        system.addForce(CMMotionRemover)

        return system
       
    def _init_langevin_integrator(self, gamma=1.0/unit.picoseconds):
        # Compute constants.
        kT = kB * self.temp
        # Create a new custom integrator.
        integrator = omm.CustomIntegrator(self.dt)
        integrator.setConstraintTolerance(1e-8)
        # Integrator initialization.
        integrator.addComputePerDof("sigma", "sqrt(kT/m)")
        integrator.addGlobalVariable("kT", kT) # thermal energy
        integrator.addGlobalVariable("T", self.temp) # temperature
        integrator.addGlobalVariable("b", np.exp(-gamma*self.dt)) # velocity mixing parameter
        integrator.addPerDofVariable("sigma", 0) 
        integrator.addPerDofVariable("x1", 0) # position before application of constraints
        # Allow context updating here.
        integrator.addUpdateContextState()
        # Velocity perturbation.
        integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        integrator.addConstrainVelocities()
        # Metropolized symplectic step.
        integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
        integrator.addConstrainVelocities()
        integrator.addComputePerDof("x", "x + v*dt")
        integrator.addComputePerDof("x1", "x")
        integrator.addConstrainPositions()
        integrator.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
        integrator.addConstrainVelocities()
        # Velocity randomization
        integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        integrator.addConstrainVelocities()
        return integrator
    
    def _init_reporters(self):
        self.simulation.reporters.append(
            HDF5Reporter(self.filename+".h5", 100))
        self.simulation.reporters.append(app.StateDataReporter(self.filename+'.log', reportInterval=100, step=True, potentialEnergy=True, separator='\t'))
        self.simulation.reporters.append(app.CheckpointReporter(self.filename+"_restart.chk",10000))

    def _init_simulation(self):
        topology = app.AmberPrmtopFile((os.path.join(self.dir, 'input/adp_implicit.prmtop'))).topology
        platform = omm.Platform.getPlatformByName('CUDA')
        simulation = app.Simulation(topology, self.system, self.integrator,platform)
        pdb = app.PDBFile((os.path.join(self.dir, 'input/adp.pdb')))
        simulation.context.setPositions(pdb.positions)
        ## minimize
        simulation.minimizeEnergy()
        return simulation
    
    def get_pe(self,position=None):
        context = self.simulation.context
        if position is not None:
            context.setPositions(position)
        state = context.getState(getEnergy = True)
        energy = state.getPotentialEnergy()
        return energy
    
    def _get_force(self,position=None):
        context = self.simulation.context
        if position is not None:
            context.setPositions(position)
        state = context.getState(getForces = True)
        force = state.getForces(asNumpy=True)
        return force  
      
    def potential(self, positions):
        energy_list = []
        for position in positions:
            position = position.detach().cpu().numpy()
            energy_list.append(self.get_pe(position=position).value_in_unit(unit.kilojoule_per_mole))
        return torch.tensor(np.array(energy_list))
    
    def save_traj_as_pdb(self,positions,dir="./"):
        topology = md.load(os.path.join(self.dir,'input/adp.pdb')).topology
        traj = md.Trajectory(xyz=positions.detach().cpu().numpy(),topology=topology)
        traj.save_pdb(dir+self.filename+".pdb")

    def force(self,positions):
        force_list = []
        for position in positions:
            position = position.detach().cpu().numpy()
            force_list.append(self._get_force(position=position).value_in_unit(
                unit.kilojoule_per_mole/unit.nanometer))
        return torch.tensor(force_list).to(positions.device)
    
    def run(self, nsteps, burnin=10000):
        if burnin>0:
            self.run(burnin)
        if self.save_traj:
            self._init_reporters()
        self.simulation.step(nsteps)
        state = self.simulation.context.getState(getEnergy = True)
        energy = state.getPotentialEnergy()
        print("end energy:",energy)
        
def main():
    parser = _parse_CLAs()
    args = parser.parse_args()
    dir = "run/"
    if not(os.path.exists(dir)):
        os.mkdir(dir)
    filename = dir+"adp"
    System = ADP(filename=filename,**vars(args))
    System.run(args.nsteps)
    System.simulation.reporters[0].close()
    traj = md.load_hdf5(filename+".h5")
    positions = traj.xyz
    print(System.potential(torch.tensor(positions)))
    #traj.save_pdb(filename+".pdb")


if __name__ == "__main__":
    main()

