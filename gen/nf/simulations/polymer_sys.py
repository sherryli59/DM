import cupy as cp
import math
import gpu_polymer as p

class Polymer():
    def __init__(self):
        self.ps = self.setup_polymer()
        self.integrator = self.setup_integrator()
    def setup_polymer(self):
        A = p.Monomer("A", 0, 1)
        B = p.Monomer("B", 0, 1)
        self.monomers = [A,B]

        monomer_dict = {}
        for monomer in self.monomers:
            if monomer.name in monomer_dict.keys():
                raise ValueError("Two monomers have the same name")
            monomer_dict[monomer.name] = monomer

        FH_terms ={
            frozenset({A}) : 0,
            frozenset({B}) : 0,
            frozenset({A, B}) : 0.6,
        }

        AB_poly = p.Polymer("AB", 40, 1, [(A, .5), (B, .5)])
        polymers = [AB_poly]

        spec_dict = {
                AB_poly :  1,
                }
        c_s = 0

        box_length = math.sqrt(10000)
        grid_spec = (2**5, 2**5)
        ps = p.PolymerSystem(self.monomers, polymers, spec_dict, FH_terms,
                box_length, grid_spec, salt_conc=c_s, integration_width = 2)
        return ps

    def setup_integrator(self):
        self.ps.get_densities()
        relax_rates = cp.ones(len(self.monomers)) * .003 * 4.5
        relax_temps = cp.ones(len(self.monomers)) * .0001 * 0

        relax_rates[-1] *= 1
        psi_relax = 0.001
        psi_temp = 0
        integration_args = (relax_rates, relax_temps, psi_relax, psi_temp)

        return p.CL_RK2(self.ps, relax_rates, relax_temps, psi_relax, psi_temp, full_epsilon=False)

    def get_energy(self, w, ps=None):
        if ps is None:
            ps = self.ps
        ps.w_all = w
        ps.update_normal_from_density()
        energy = ps.get_free_energy()
        return energy

    def get_density(self, w, ps=None):
        if ps is None:
            ps = self.ps
        ps.w_all = w
        ps.update_normal_from_density()
        ps.get_densities()
        density = ps.phi_all
        return density

    def run_dynamics(self, w, integrator=None, burn_steps=100, run_steps=900):
        if integrator is None:
            integrator = self.integrator
        integrator.ps.w_all = w
        integrator.ps.update_normal_from_density()

        integrator.relax_rates *= 0.1
        for _ in range(burn_steps):
            integrator.complex_langevin()
        integrator.relax_rates *= 10
        for _ in range(run_steps):
            integrator.complex_langevin()
        integrator.ps.update_density_from_normal()
        return integrator.ps.w_all