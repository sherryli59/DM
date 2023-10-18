from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import numpy as np
from mdtraj.reporters import HDF5Reporter
import argparse

class SingleLipid():
    def __init__(self, base_filename="test", temperature=323.15,
                 friction=1, dt=0.002, integrator_to_use="ovrvo",
                 folder_name="./", platform="CUDA",
                 ff_kwargs=dict(nonbondedMethod=PME,
                                nonbondedCutoff=1.2 * nanometer,
                                constraints=HBonds,
                                switchDistance=0.8 * nanometer)):
        
        self.folder_name = folder_name
        self.dir = os.path.realpath(os.path.dirname(__file__))
        psf = CharmmPsfFile(base_filename + ".psf")
        pdb = PDBFile(base_filename + ".pdb")
        x_dim = (max([x.x for x in pdb.positions])
                 - min([x.x for x in pdb.positions]))
        y_dim = (max([x.y for x in pdb.positions])
                 - min([x.y for x in pdb.positions]))
        z_dim = (max([x.z for x in pdb.positions])
                 - min([x.z for x in pdb.positions]))
        dry_box_dim = np.array([x_dim, y_dim, z_dim]) * 10
        dry_topology = psf.topology
        dry_positions = pdb.positions
        ff = ForceField(os.path.join(self.dir, "charmm_files/custom_xml/charmm36_custom_all.xml"))
        temperature = temperature * kelvin
        friction /= picoseconds
        dt = dt * picoseconds

        self.beta = 1/(temperature * BOLTZMANN_CONSTANT_kB *
                       AVOGADRO_CONSTANT_NA)
        dry_system = self._init_system(topology=dry_topology,
                                       box_dim=dry_box_dim, ff=ff)


        platform = Platform.getPlatformByName(platform)
        if integrator_to_use == "ovrvo":
            integrator = self._get_ovrvo_integrator(temperature, friction, dt)
        elif integrator_to_use == "verlet":
            integrator = self._get_verlet_integrator(temperature, friction, dt)
        elif integrator_to_use == "overdamped":
            integrator = self._get_overdamped_integrator(temperature, friction, dt)
        else:
            raise ValueError("Incorrect integrator supplied")
        


        platform = Platform.getPlatformByName('CUDA')
        self.simulation = Simulation(dry_topology, dry_system,
                                     integrator, platform)
        self.simulation.context.setPositions(dry_positions)
        self.simulation.minimizeEnergy()
        self.target_atom_indices = self._get_target_atom_indices()
        reporter = HDF5Reporter(self.folder_name + "sim.h5", 1000, atomSubset=self.target_atom_indices)
        self.simulation.reporters.append(reporter)
        self.simulation.context.setVelocitiesToTemperature(temperature)

        
    def _get_overdamped_integrator(self, temperature, friction, dt):
        """Creates OpenMM integrator to carry out overdamped Brownian integration
        Arguments:
            temperature: temperature with OpenMM units
            friction: friction coefficient with OpenMM units
            dt: time step with OpenMM units
        Returns:
            overdamped_integrator: OpenMM Integrator
        """

        overdamped_integrator = CustomIntegrator(dt)
        overdamped_integrator.addGlobalVariable("kT", 1/self.beta)
        overdamped_integrator.addGlobalVariable("friction", friction)

        overdamped_integrator.addUpdateContextState()
        overdamped_integrator.addComputePerDof("x", "x+dt*f/(m*friction) + gaussian*sqrt(2*kT*dt/(m*friction))")
        return overdamped_integrator
    
    def _get_ovrvo_integrator(self, temperature, friction, dt):
        """Creates OpenMM integrator to carry out ovrvo integration (Sivak, Chodera, and Crooks 2014)
        Arguments:
            temperature: temperature with OpenMM units
            friction: friction coefficient with OpenMM units
            dt: time step with OpenMM units
        Returns:
            ovrvo_integrator: OpenMM Integrator
        """
        ovrvo_integrator = CustomIntegrator(dt)
        ovrvo_integrator.setConstraintTolerance(1e-8)
        ovrvo_integrator.addGlobalVariable("a", math.exp(-friction * dt/2))
        ovrvo_integrator.addGlobalVariable(
            "b", np.sqrt(1 - np.exp(-2 * friction * dt/2)))
        ovrvo_integrator.addGlobalVariable("kT", 1/self.beta)
        ovrvo_integrator.addPerDofVariable("x1", 0)

        ovrvo_integrator.addComputePerDof(
            "v", "(a * v) + (b * sqrt(kT/m) * gaussian)")
        ovrvo_integrator.addConstrainVelocities()

        ovrvo_integrator.addComputePerDof("v", "v + 0.5*dt*(f/m)")
        ovrvo_integrator.addConstrainVelocities()

        ovrvo_integrator.addComputePerDof("x", "x + dt*v")
        ovrvo_integrator.addComputePerDof("x1", "x")
        ovrvo_integrator.addConstrainPositions()
        ovrvo_integrator.addComputePerDof("v", "v + (x-x1)/dt")
        ovrvo_integrator.addConstrainVelocities()
        ovrvo_integrator.addUpdateContextState()

        ovrvo_integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
        ovrvo_integrator.addConstrainVelocities()
        ovrvo_integrator.addComputePerDof(
            "v", "(a * v) + (b * sqrt(kT/m) * gaussian)")
        ovrvo_integrator.addConstrainVelocities()
        return ovrvo_integrator
    
    def _get_verlet_integrator(self, temperature, friction, dt):
        """Creates OpenMM integrator to carry out Verlet integration
        Arguments:
            temperature: temperature with OpenMM units
            friction: friction coefficient with OpenMM units
            dt: time step with OpenMM units
        Returns:
            verlet_integrator: OpenMM Integrator
        """

        verlet_integrator = CustomIntegrator(dt)
        verlet_integrator.addUpdateContextState()
        verlet_integrator.addComputePerDof("v", "v+0.5*dt*f/m")
        verlet_integrator.addComputePerDof("x", "x+dt*v")
        verlet_integrator.addComputePerDof("v", "v+0.5*dt*f/m")
        return verlet_integrator


    def _get_target_atom_indices(self):
        """Gets the indices of all non H2O atoms
        Returns:
            all_atom_indices: The indices of all non water atoms
        """
        all_atom_indices = []
        for residue in self.simulation.topology.residues():
            if residue.name != "HOH":
                for atom in residue.atoms():
                    all_atom_indices.append(atom.index)
        return all_atom_indices    


    def _init_system(self, topology, box_dim, ff,
                     ff_kwargs=dict(nonbondedMethod=PME,
                                    nonbondedCutoff=1.2 * nanometer,
                                    constraints=HBonds,
                                    switchDistance=0.8 * nanometer)):
        """Creates a system from a given topology, box_dim and force field
        Arguments:
            topology: An OpenMM topology for the system
            box_dim: A numpy array of shape (3) corresponding to box dimensions
            ff: OpenMM ForceField required to make the system
            ff_kwargs: key word arguments required to create syste,
        Returns:
            system: An OpenMM system corresponding to the syste
        """
        topology.setPeriodicBoxVectors([Vec3(box_dim[0], 0, 0),
                                        Vec3(0, box_dim[1], 0),
                                        Vec3(0, 0, box_dim[2])
                                        ])
        system = ff.createSystem(topology, **ff_kwargs)
        return system

    def run_sim(self, steps, close_file=False):
        """Runs self.simulation for steps steps
        Arguments:
            steps: The number of steps to run the simulation for
            close_file: A bool to determine whether to close file. Necessary
            if using HDF5Reporter
        """
        self.simulation.step(steps)
        if close_file:
            self.simulation.reporters[0].close()

    def get_energy(self, positions, enforce_periodic_box=False):
        """Updates position and velocities of the system
        Arguments:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in Angstroms
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in Angstroms/ps
        """
        positions = positions * nanometers
        self.simulation.context.setPositions(positions)
        state = self.simulation.context.getState(getPositions=False,
                                                 getEnergy=True,
                                                 enforcePeriodicBox=enforce_periodic_box)
        pe = state.getPotentialEnergy() * self.beta
        return pe
        

    def get_information(self, as_numpy=True, enforce_periodic_box=False):
        """Gets information (positions, forces and PE of system)
        Arguments:
            as_numpy: A boolean of whether to return as a numpy array
            enforce_periodic_box: A boolean of whether to enforce periodic boundary conditions
        Returns:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in Angstroms
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in Angstroms/ps
            forces: A numpy array of shape (n_atoms, 3) corresponding to the force in kcal/mol*Angstroms
            pe: A float coressponding to the potential energy in kcal/mol
            ke: A float coressponding to the kinetic energy in kcal/mol
        """
        state = self.simulation.context.getState(getForces=True,
                                                 getEnergy=True,
                                                 getPositions=True,
                                                 getVelocities=True,
                                                 enforcePeriodicBox=enforce_periodic_box)
        positions = state.getPositions(asNumpy=as_numpy).in_units_of(nanometers)
        forces = state.getForces(asNumpy=as_numpy).in_units_of(kilojoules_per_mole / nanometers)
        velocities = state.getVelocities(asNumpy=as_numpy).in_units_of(nanometers / picoseconds)
        positions = positions._value
        forces = forces._value
        velocities = velocities._value
        
        pe = state.getPotentialEnergy().in_units_of(kilojoules_per_mole)._value
        pe_beta = pe * self.beta
        ke = state.getKineticEnergy().in_units_of(kilojoules_per_mole)._value

        return positions, velocities, forces, pe, ke, pe_beta
    

    def relax_position(self, init_pos, num_data_points=int(1E6), save_freq=250, tag = ""):

        init_pos = init_pos * nanometers
        self.simulation.context.setPositions(init_pos)
        all_positions = []
        all_velocities = []
        all_forces = []
        all_pe = []
        all_pe_beta = []
        all_ke = []


        positions, velocities, forces, pe, ke, pe_beta = self.get_information(enforce_periodic_box=False)
        all_positions.append(positions)
        all_velocities.append(velocities)
        all_forces.append(forces)
        all_pe.append(pe)
        all_pe_beta.append(pe_beta)
        all_ke.append(ke)


        for i in range(num_data_points):
            self.run_sim(save_freq)
            positions, velocities, forces, pe, ke, pe_beta = self.get_information(enforce_periodic_box=False)
            all_positions.append(positions)
            all_velocities.append(velocities)
            all_forces.append(forces)
            all_pe.append(pe)
            all_pe_beta.append(pe_beta)
            all_ke.append(ke)

            if (i % 1000 == 0 or i == (num_data_points - 1)):
                np.save(self.folder_name + tag + "positions.npy", all_positions)
                np.save(self.folder_name + tag + "velocities.npy", all_velocities)
                np.save(self.folder_name + tag + "forces.npy", all_forces)
                np.save(self.folder_name + tag + "pe.npy", all_pe)
                np.save(self.folder_name + tag + "pe_beta.npy", all_pe_beta)
                np.save(self.folder_name + tag + "ke.npy", all_ke)
        
        np.save(self.folder_name + tag + "positions.npy", all_positions)
        np.save(self.folder_name + tag + "velocities.npy", all_velocities)
        np.save(self.folder_name + tag + "forces.npy", all_forces)
        np.save(self.folder_name + tag + "pe.npy", all_pe)
        np.save(self.folder_name + tag + "pe_beta.npy", all_pe_beta)
        np.save(self.folder_name + tag + "ke.npy", all_ke)


    
    def generate_long_trajectory(self, num_data_points=int(1E8), save_freq=1000, tag = ""):
        """Generates long trajectory of length num_data_points*save_freq time steps where information (pos, vel, forces, pe, ke)
           are saved every save_freq time steps
        Arguments:
            num_data_points: An int representing the number of data points to generate
            save_freq: An int representing the frequency for which to save data points
            tag: A string representing the prefix to add to a file
        Saves:
            tag + "_positions.txt": A numpy array of shape (num_data_points*n_atoms,3) representing the positions of the trajectory in units of Angstroms
            tag + "_velocities.txt": A numpy array of shape (num_data_points*n_atoms,3) representing the velocities of the trajectory in units of  Angstroms/picoseconds
            tag + "_forces.txt": A numpy array of shape (num_data_points*n_atoms,3) representing the forces of the trajectory in units of kcal/mol*Angstroms
            tag + "_pe.txt": A numpy array of shape (num_data_points,) representing the pe of the trajectory in units of kcal/mol
            tag + "_ke.txt": A numpy array of shape (num_data_points,) representing the ke of the trajectory in units of kcal/mol
        """

        for _ in range(num_data_points):
            self.run_sim(save_freq)
     

def _parse_CLAs():
    parser = argparse.ArgumentParser(description="Run a \
    simulation of Lennard Jones liquid")
    parser.add_argument('--replicate', type=int, default=0, help='index of \
    statistical replicate.')
    parser.add_argument("--lipid_type", type=str, default="dlpc",
                    choices=["dlpc", "dppc", "dmpc", "dopc", "pope"])
    parser.add_argument('--temp', type=float, default=323.15, help='Temperature\
    defaults to 323.15')
    parser.add_argument('--output', type=str, default='output/', help='Where to \
    save the output of the simulation')
    parser.add_argument('--nsteps', type=int, default=100000, help='How many \
    samples to collect')
    parser.add_argument('--burnin', type=int, default=100000, help='Number of \
    steps to burn in the trajectory for.')
    parser.add_argument('--platform', type=str, default="CUDA", help='Default platform \
    is CUDA.')
    return parser

def main():
    parser = _parse_CLAs()
    args = parser.parse_args()
    if not(os.path.exists(args.output)):
        os.mkdir(args.output)
    dir = args.output+"lipid_%s/"%args.lipid_type
    temperature_pairs = dict(dlpc=303,
                         dppc=323.15,
                         dopc=296,
                         dmpc=303,
                         popc=303, pope=310.15)
    temp = temperature_pairs[args.lipid_type]
    if not(os.path.exists(dir)):
        os.mkdir(dir)
    dir = dir + "run%d/"%args.replicate
    if not(os.path.exists(dir)):
        os.mkdir(dir)
    sys = SingleLipid(base_filename="input/"+args.lipid_type, folder_name=dir, temperature=temp, platform=args.platform)
    sys.generate_long_trajectory(args.nsteps,tag=args.lipid_type)

if __name__ == "__main__":
    main()


