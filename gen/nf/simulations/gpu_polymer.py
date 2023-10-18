import math
import numpy as np
import cupy as cp 
import cupyx.scipy.fft as cufft
from scipy import ndimage
import warnings



class Grid(object):
    """docstring for Grid.

    grid_spec = (n_x, n_y, n_z)

    generalizable, but 2d for now

    """

    def __init__(self, box_length=4., grid_spec=(2**6, 2**6)):
        super(Grid, self).__init__()

        self.grid_spec = grid_spec
        self.ndims = len(self.grid_spec)
        self.l = box_length
        self.update_l(self.l)


    def update_l(self, new_l):
        self.l = new_l

        self.V = self.l**self.ndims
        self.grid = cp.asarray(cp.meshgrid(
            *[cp.linspace(0, self.l, n) for n in self.grid_spec]))
        self.kgrid = cp.asarray(cp.meshgrid(*[2 * cp.pi / self.l 
            * cp.concatenate((cp.arange(0, n / 2 + 1),
                cp.arange(-n / 2 + 1, 0)), axis=None) for n in self.grid_spec]))
        self.k1 = cp.sum(self.kgrid, axis=0)
        self.k2 = cp.sum(self.kgrid**2, axis=0)
        self.dV = self.V / self.k2.size

        return
class Monomer(object):
    """
    Going to be used to store the properties of all the monomers in the system
    """
    def __init__(self, name, charge, epsilon, seg_len = 1, identity='solvent',
            has_volume=True, alpha=1):
        
        self.name = name
        self.alpha = alpha
        self.charge = charge
        self.epsilon = epsilon
        self.has_volume = has_volume
        self.seg_len = seg_len
        self.identity = identity

    def __repr__(self):
        return self.name

class Polymer(object):
    """docstring for Polymer.
    block_structure = [("A", f_A), ("B", f_B)]
    """

    def __init__(self, name, n_monomers, segment_length, block_structure, 
            has_volume=True,):
        super(Polymer, self).__init__()
        self.name = name
        self.n_monomers = n_monomers
        self.block_structure = block_structure
        self.identity = 'polymer_string'
        
        self.struct = None
        
        for monomer in set(p[0] for p in self.block_structure):
            if monomer.identity != 'polymer':
                monomer.identity = 'polymer'
        
    def __repr__(self):
        return str(self.block_structure)

    def build_working_polymer(self, h):
        if self.struct is not None:
            raise ValueError("polymer structure should only be built once")
        total_h = self.n_monomers 
        hold_struct = [] 
        hold_h_struct = []
        where = 0. 
        end = 0. 
        
        for name_tuple in self.block_structure:
            end += name_tuple[1] * total_h
            while where < end: 
                if where + h > end:
                    hold_struct.append(name_tuple[0])
                    hold_h_struct.append(end - where)
                    where = end
                    continue
                hold_struct.append(name_tuple[0])
                hold_h_struct.append(h)
                where += h

        self.struct = np.asarray(hold_struct)
        self.h_struct = cp.asarray(hold_h_struct, dtype='float64')
        return 

class PolymerSystem(object):
    """docstring for PolymerSystem."""

    #TODO: store chi_ij early on?
    def __init__(self, monomers, polymers, spec_dict, FH_dict, box_length,
            grid_spec, salt_conc=0., integration_width=4,
            salt=True, custom_salts=None):
        super(PolymerSystem, self).__init__()
        self.n_species = len(monomers)
        self.grid_spec = grid_spec
        self.integration_width = integration_width
        self.FH_dict = FH_dict

        #we want the last monomer species to be solvent if possible
        self.set_monomer_order(monomers)

        #sort the species dictionary into polymer and solvent components
        self.polymers = polymers
        self.poly_dict = {}
        self.solvent_dict = {}
        self.Q_dict = {}
        check_frac = 0


        #TODO: REMOVE LATER:
        self.t1 = 0 
        for spec in spec_dict.keys():
            check_frac += spec_dict[spec]
            if spec.__class__.__name__ == "Polymer":
                self.poly_dict[spec] = spec_dict[spec]
            elif spec.__class__.__name__ == "Monomer":
                self.solvent_dict[spec] = spec_dict[spec]
            else:
                raise ValueError("Unknown member of species dictionary")
        #check that the total fraction sums to one 
        if not math.isclose(check_frac, 1.):
            raise ValueError("Total volume fraction must sum to 1")
       

        #build flory huggins matrix 
        self.FH_matrix = cp.zeros((self.n_species, self.n_species))
        
        for i in range(len(self.monomers)):
            for j in range(len(self.monomers)):
                self.FH_matrix[i,j] = \
                        self.FH_dict[frozenset((monomers[i], monomers[j]))]

        #write the actual integration frameworks to each polymer
        for polymer in self.poly_dict:
            polymer.build_working_polymer(self.integration_width)
        
        #Set up grid
        self.grid = Grid(box_length, grid_spec)

        #Check for degeneracies in representation and correct if necessary
        self.find_degeneracy()
        #Set up normal transforms
        self.generate_effective_FH()
        self.generate_density_coeffs()
        self.assign_normals()
        self.get_gamma()

        #Initialize all fields (currently to zero)
        self.w_all = cp.zeros([self.red_FH_mat.shape[0]] + list(self.grid.k2.shape))
        self.psi = cp.zeros_like(self.grid.k2)
        self.w_all = self.randomize_array(self.w_all, 0)

        #Initialize mu field
        self.update_normal_from_density()

        #TODO:REMOVE THIS
        self.smear_const = 0.00012

        #check if we want to include salts
        if abs(salt_conc) == 0:
            salt = False
        if salt is False: 
            self.use_salts = False
            return 
        self.use_salts = True

        #initialize salts
        self.c_s = salt_conc
        if custom_salts is None:
            self.salt_pos = Monomer("salt+", 1, 1,
                    identity='salt', has_volume=False)
            self.salt_neg = Monomer("salt-", -1, 1,
                    identity='salt', has_volume=False)
            self.salts = (self.salt_pos, self.salt_neg)
        else:
            self.salts = custom_salts
        if len(self.salts) not in (0, 2):
            raise NotImplentedError("Unusual number of salts")
        self.salt_concs = cp.zeros(len(self.salts))

        return

        
    def find_degeneracy(self):
        #Only works if two components have identical FH parameters (for now)
        #TODO: probably can rewrite this to handle cases where two parameters
        # are scaled or linear combinations but that would require more work

        #WARNING: IS NOT GUARANTEED TO REMOVE ALL DEGENERACIES JUST EASY ONES
        degen_sets = []
        #identify degeneracy
        for i in range(self.FH_matrix.shape[0]):
            for j in range(i+1, self.FH_matrix.shape[0]):
                if np.allclose(self.FH_matrix[i], self.FH_matrix[j]):
                    degen_sets.append({i,j})
        reducing=True

        #Horrible code to combine the degeneracies
        while reducing:
            reducing=False
            return_to_outer_loop=False
            for i in range(len(degen_sets)):
                if return_to_outer_loop==True:
                    break
                for j in range(i+1, len(degen_sets)):
                    if len(degen_sets[i].union(degen_sets[j])) != \
                            len(degen_sets[i]) + len(degen_sets[j]):
                        return_to_outer_loop=True
                        reducing=True
                        degen_sets.append(degen_sets[i].union(degen_sets[j]))
                        degen_sets.pop(i)
                        degen_sets.pop(j)
                        break
                        
        degen_lists = [sorted(x) for x in degen_sets]
        #generate new non-degenerate matrix:
        mask = np.ones(self.FH_matrix.shape[0], bool)
        #generate non-degenerate FH matrix
        for x in degen_lists:
            mask[x[1:]] = 0
        kept_indices = np.arange(len(mask))[mask]
        self.red_FH_mat = self.FH_matrix[kept_indices][:,kept_indices]
        #write a dictionary to record the new indices of the FH matrix to the 
        # original species
        self.degen_dict = {}
        self.rev_degen_dict = {}
        for i in range(kept_indices.size):
            modified=False
            for degen in degen_lists:
                if kept_indices[i] in degen:
                    modified=True
                    self.degen_dict[i] = [self.monomers[k] for k in degen]
                    for j in degen:
                        self.rev_degen_dict[self.monomers[j]] = i
            if modified==False:
                self.degen_dict[i] = [self.monomers[kept_indices[i]]]
                self.rev_degen_dict[self.monomers[kept_indices[i]]] = i
        return

    def reduce_phi_all(self, phi_all):
        #Helper function to convert densities into the densities when 
        # degenerate species modes were summed over 
        red_phi_all = cp.zeros([len(self.degen_dict)] + list(phi_all.shape[1:]))
        for i in range(red_phi_all.shape[0]):
            for mon in self.degen_dict[i]:
                red_phi_all[i] += phi_all[self.monomers.index(mon)]
        return red_phi_all



    def generate_effective_FH(self):
        #Function to build the X matrix out of the FH terms 
        #(see Duchs and Fredrickson 2014)
        #assumes that the last term is always the solvent term
        X = cp.zeros([x-1 for x in self.red_FH_mat.shape])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X[i,j] = self.red_FH_mat[i,j]
        for i in range(X.shape[0]):
            X[i,:] += (-2) * self.red_FH_mat[i,-1]
        X = (X + X.T)/2
        self.effective_FH = X
       
        return

    def generate_density_coeffs(self):
        #get coefficients that augment effective_FH matrix
        # these are \chi_jS 
        h_s = cp.zeros(self.red_FH_mat.shape[0]-1)
        for i in range(len(h_s)):
            h_s[i] = self.red_FH_mat[i,-1]

        self.density_coeffs = h_s

        return 

    def set_monomer_order(self, monomers):
        #permanently affix the monomer order in the form of a tuple
        forward_count = 0
        reverse_count = len(monomers) - 1
        temp_list = monomers.copy()
        for monomer in monomers: 
            if monomer.identity == 'solvent':
                temp_list[reverse_count] = monomer
                reverse_count -= 1
            else:
                temp_list[forward_count] = monomer
                forward_count += 1
        self.monomers = tuple(temp_list)
        return 

    def get_gamma(self):
        #determine which fields are real and imaginary and assign correct gamma
        gamma = cp.zeros(self.normal_evalues.size + 1, dtype='complex128')
        gamma += 1j * (cp.pad(self.normal_evalues, ([0,1]), mode='constant',
            constant_values=(1)) > 0)
        gamma += cp.logical_not((cp.pad(self.normal_evalues, ([0,1]),
            mode='constant', constant_values=(1)) > 0))
        self.gamma = gamma
        return

    def assign_normals(self):
        #assign coefficients for normal mode tranform
        self.normal_evalues, self.normal_modes = \
        cp.linalg.eigh(self.effective_FH)
        warning_thresh = 1e-3
        if cp.amin(cp.abs(self.normal_evalues)) <= warning_thresh:
            danger = cp.amin(cp.abs(self.normal_evalues))
            warnings.warn("Minimum eigenvalue is " \
                    + "{:.3}".format(danger) \
                    + " which is very small and likely to cause problems")
        
        self.A_ij = A = cp.pad(self.normal_modes, ([0,1]), mode='constant')
        self.A_ij[:,-1] = 1
        self.A_inv = cp.linalg.inv(self.A_ij)

        return 

    def randomize_array(self, array, noise): 
        #randomize given array, used for initialization 
        array = cp.random.random_sample(size = array.shape) * noise
        return array

    def get_epsilon(self):
        #get the epsilon from a given density configuration
        monomer_epsilons = cp.asarray([mon.epsilon for mon in self.monomers])
        self.tot_epsilon = cp.sum((monomer_epsilons*self.phi_all.T).T \
                / cp.sum(self.phi_all, axis=0),axis=0)

        return 

    def update_normal_from_density(self):
        #update the normal mode representation to match current real
        #represenation 
        
        self.normal_w = cp.reshape(
                self.A_inv@cp.reshape(self.w_all, (self.w_all.shape[0],-1)),
                (self.w_all.shape))
        return 

    def update_density_from_normal(self):
        #update the real representation to match current normal mode
        #represenation 
        self.w_all = cp.reshape(
                self.A_ij@cp.reshape(self.normal_w, (self.normal_w.shape[0],-1)),
                (self.normal_w.shape))

        return 

    def set_field_averages(self):
        #TODO: Determine if this works
        #Experimental, idea is to analytically set the field averages to obey 
        #symmetry and hopefully allow the Gibbs ensemble to work
        
        axes = np.arange(len(self.normal_w.shape))[1:]
        red_phi = self.reduce_phi_all(self.phi_all)
        average_phi = np.average(red_phi, axis=axes)

        #calculate the analytically expected averages for all the fields
        avg_w_norm = np.zeros(self.red_FH_mat.shape[0])
        for i in range(self.red_FH_mat.shape[0]-1): 
            for j in range(self.red_FH_mat.shape[1]-1):
                chi_jS = self.FH_dict[frozenset({self.monomers[j], 
                    self.monomers[-1]})]
                avg_w_norm[i] += self.normal_modes[j,i] * self.density_coeffs[j]
                avg_w_norm[i] += average_phi[j] * self.normal_modes[j,i] \
                        * self.normal_evalues[i]
        
        #adjust them to hopefully respect symmetry
        avg_w = self.A_ij.get()@avg_w_norm
        avg_w_norm = cp.asarray(avg_w_norm)
        w_shift = np.full_like(avg_w, -np.average(avg_w))
        target_w = avg_w + w_shift
        target_w_norm = self.A_inv.get()@target_w
        current_average =  cp.asarray(np.average(self.normal_w, axis=axes))
        self.update_density_from_normal()
        self.normal_w = (self.normal_w.T + (cp.asarray(target_w_norm) - current_average).T).T
        self.update_density_from_normal()

    def gradient(self, array):
        #internal gradient for species on the grid
        array_k = cufft.fftn(array, s=array.shape)
        #array_k = cp.fft.fftn(array, s=array.shape)
        array_k.flat[0] = 0 #check if this is reasonable
        #grad_array_k = array_k * cp.exp((0 + 1j)* self.grid.k1)
        exp_mult_comp= cp.ElementwiseKernel(
                'complex128 q_k, complex128 im_K1', 
                'complex128 out', 
                '''
                out = q_k * exp(im_K1);
                ''',
                'exp_mult_comp', 
                preamble='#include <cupy/complex.cuh>')
        grad_array_k = exp_mult_comp(array_k, (1j)*self.grid.k1) 
        grad_array = cufft.irfftn(grad_array_k, s=array.shape)

        return grad_array

    def laplacian(self, array):
        #internal gradient for species on the grid
        array_k = cufft.fftn(array, s=array.shape)
        lap_array_k = array_k * -self.grid.k2
        lap_array = cufft.irfftn(lap_array_k, s=array.shape)

        return lap_array
    def gaussian_smear(self, array, alpha):
        #Smear a given array by a gaussian with given variance and zero mean

        fourier_gauss = cp.exp(-math.pi**2 * self.grid.k2 * alpha) 
        #/ cp.sqrt(2 * math.pi)
#        fourier_gauss /= cp.sum(fourier_gauss) #* cp.sqrt(math.pi / alpha)
        
        kernel_mult= cp.ElementwiseKernel(
                'complex128 q_k, float64 kernel', 
                'complex128 out', 
                'out = q_k * kernel',
                'kernel_mult')
    
        array_k = cufft.fftn(array, s=array.shape)
        array_k = kernel_mult(array_k, fourier_gauss)
        array_r = cufft.irfftn(array_k, s=array_k.shape)
        #if abs(cp.sum(array_r)) > 1e-8:
        #    array_r *= cp.sum(array) / cp.sum(array_r)
        return array_r

    def get_total_charge(self, include_salt=True):
        #gets the total charge at every grid point
        total_charge = cp.zeros((self.phi_all[0].shape))
        for i in range(len(self.monomers)):
            
            total_charge += self.monomers[i].charge * self.monomers[i].alpha *\
                    self.phi_all[i]
        if include_salt==False:
            return total_charge
        
        if self.use_salts == False:
            return total_charge
        for i in range(len(self.salts)):
            total_charge += self.salts[i].charge * self.phi_salt[i]
        return total_charge

    def get_densities(self, *args, get_phi_del=False):
        try: 
            self.get_phi_del
        except AttributeError:
            self.get_phi_del = False

        if get_phi_del is True:
            self.get_phi_del = True
        #Function to get the phis by solving the modified diffusion equation 
        q_r0 = cp.ones_like(self.w_all[0])
        q_r_dag0 = cp.ones_like(self.w_all[0])
        self.phi_all = cp.zeros([len(self.rev_degen_dict)] + list(self.grid.k2.shape))
   
        #TODO: DELETE THIS
        #build P_species from w_all and poly_list
        P_species = {}
        for monomer in self.monomers:
            if monomer.has_volume:
                #effective field from total of potentials
                #TODO: FIX TESTING TESTING GAUSSIAN SMEARING
                P_species[monomer] = self.gaussian_smear(self.w_all[self.rev_degen_dict[monomer]] \
                - self.psi * monomer.alpha *\
                monomer.charge \
                / (monomer.seg_len**self.grid.ndims), self.smear_const) 
                #P_species[monomer] = self.w_all[self.rev_degen_dict[monomer]] \
                #- self.psi * monomer.alpha *\
                #monomer.charge \
                #/ (monomer.seg_len**self.grid.ndims) 
                
        #subroutine to collect special values for osmotic pressure
        if self.get_phi_del:
            self.phi_del_dict = {}
        elif 'phi_del_dict' in self.__dict__:
            del self.phi_del_dict

        #Iterate over all polymer types
        for polymer in self.poly_dict:
            f_poly = self.poly_dict[polymer]
            if f_poly == 0:
                continue
    
            #step size along polymer 
            q_r_s, q_r_dag_s = integrate_s(polymer.struct, 
                    polymer.h_struct, P_species, q_r0, q_r_dag0, self.grid.k2)
            cp.cuda.stream.get_current_stream().synchronize()
            #Partition function as a function of s
            Q_c = q_r_dag_s * q_r_s
            Q_c = self.reindex_Q_c(Q_c)

            #partition function across entire polymer
            Q = cp.sum((Q_c)[-1]) * self.grid.dV / self.grid.V
            
            #collecting q_dag*lap(q) for osmotic pressure
            if self.get_phi_del is True:
                lap_q_r_s = cp.zeros_like(q_r_s)
                for i in range(lap_q_r_s.shape[0]):
                    lap_q_r_s[i] = self.laplacian(q_r_s[i])
                
                Q_del_c = q_r_dag_s * lap_q_r_s
                Q_del_c = self.reindex_Q_c(Q_del_c)

                phi_del = cp.sum((Q_del_c.T * polymer.h_struct).T, axis=0)\
                        * f_poly / (Q * cp.sum(polymer.h_struct))

                self.phi_del_dict[polymer] = phi_del
            #check that Q is equal across integral (necessary condition)
            if not cp.allclose(cp.sum(Q_c, axis=tuple(range(1,len(Q_c.shape)))) * self.grid.dV /\
                    self.grid.V, Q):
                print(Q_c)
                raise ValueError("Q_c not equal across integral")

            self.Q_dict[polymer] = cp.copy(Q)

            #generate phi's by summing over partition function in correct areas 
            for i in range(len(self.monomers)):
                self.phi_all[i] += cp.sum((Q_c.T * polymer.h_struct).T\
                        [polymer.struct==self.monomers[i]],axis=0)\
                        * f_poly / (Q * cp.sum(polymer.h_struct)) 

            cp.cuda.stream.get_current_stream().synchronize()
        phi_salt_shape = list(self.w_all.shape)
        phi_salt_shape[0] = 2
        self.phi_salt = cp.zeros(phi_salt_shape)
   
        #compute solvent densities
        for solvent in self.solvent_dict:
                idx = self.monomers.index(solvent)
                #exp_w_S = cp.exp(-P_species[self.monomers[i]])
                #TESTING GAUSSIAN SMEARING
                exp_w_S = cp.exp(-self.gaussian_smear(P_species[self.monomers[i]], self.smear_const))
                Q_S = cp.sum(exp_w_S) /(self.grid.k2.size)
                self.phi_all[idx] = (exp_w_S * self.solvent_dict[solvent]/ Q_S)
                self.Q_dict[solvent] = cp.copy(Q_S) 
        
        #update epsilon whenever we update density 
        self.get_epsilon()

        #check if we are using salts
        if self.use_salts==False:
            return

        #need to find the correct charge for each of the salt species
        net_saltless_charge = cp.sum(self.get_total_charge(include_salt=False))

        salt_charges = [salt.charge for salt in self.salts]
        if net_saltless_charge < self.c_s * self.grid.k2.size *\
                min(salt_charges)\
        or net_saltless_charge > self.c_s * self.grid.k2.size *\
                max(salt_charges):
            print("Salt needed is ", abs(net_saltless_charge)/\
                    (self.grid.k2.size))
            raise ValueError("Inadequate salt to correct charge imbalance")
        for i in range(len(self.salts)):
            if self.c_s == 0:
                break
            self.salt_concs[i] = \
            salt_conc = (self.c_s - net_saltless_charge /\
            (self.salts[i].charge * self.grid.k2.size)) / 2
                        
            w_salt = self.salts[i].charge * -self.psi 
            exp_w_salt = cp.exp(-w_salt) 
            Q_salt = cp.sum(exp_w_salt) /(self.grid.k2.size)
            self.phi_salt[i] = (exp_w_salt * salt_conc / Q_salt)
            self.Q_dict[self.salts[i]] = Q_salt

        
        if self.get_phi_del is True:
            self.get_phi_del = False
        return 

    def reindex_Q_c(self, Q_c):
        #to correctly handle the partition function the points should be
        #associated with the edges, but to get the weighting for the polymers
        #we need to measure at the interpolation, this function reindexes and 
        #resamples Q_c to associate the partition functions with the beads
        shape = list(Q_c.shape)
        shape[0] -= 1
        new_Q_c = cp.zeros(shape)
        new_Q_c+= Q_c[1:]/2 + Q_c[:-1]/2
    
        return new_Q_c   

    def get_structure_factor(self):

        total_charge = self.get_total_charge()

        self.structure_fact = cp.zeros_like(self.phi_all)
        self.structure_fact_salt = cp.zeros_like(self.phi_salt)
        self.structure_fact_charge = cp.zeros_like(total_charge)
        wheres = cp.unique(self.grid.k2)[1::]
        self.s_fact_1d = cp.zeros((self.phi_all.shape[0], wheres.size))
        self.s_fact_1d_salt = cp.zeros((self.phi_all.shape[0], wheres.size))
        self.s_fact_1d_charge = cp.zeros(wheres.size)
        
        for i in range(self.phi_all.shape[0]):
            phi_k = cufft.fftn(self.phi_all[i], s=self.phi_all[i].shape)
            self.structure_fact[i] = cp.abs(phi_k)**2
            self.structure_fact[i].flat[0] = 0
            for j in range(self.s_fact_1d.shape[-1]):
                self.s_fact_1d[i,j] = \
                        cp.average(self.structure_fact[i,self.grid.k2==wheres[j]]) 
       
        for i in range(self.phi_salt.shape[0]):
            phi_k = cufft.fftn(self.phi_salt[i], s=self.phi_salt[i].shape)
            self.structure_fact_salt[i] = cp.abs(phi_k)**2
            self.structure_fact_salt[i].flat[0] = 0
            for j in range(self.s_fact_1d.shape[-1]):
                self.s_fact_1d_salt[i,j] = \
                        cp.average(self.structure_fact_salt[i,self.grid.k2==wheres[j]]) 


        c_k = cufft.fftn(total_charge, s=total_charge.shape)
        self.structure_fact_charge = cp.abs(c_k)**2
        self.structure_fact_charge.flat[0] = 0
        for j in range(self.s_fact_1d.shape[-1]):
            self.s_fact_1d_charge[j] = \
                    cp.average(self.structure_fact_charge[self.grid.k2==wheres[j]]) 
        #recenter the structure factor?
        self.structure_fact = cp.roll(self.structure_fact,
                [x//2 for x in self.structure_fact.shape[1:]],
                axis=range(1, self.structure_fact.ndim))
        self.structure_fact_salt = cp.roll(self.structure_fact_salt,
                [x//2 for x in self.structure_fact_salt.shape[1:]],
                axis=range(1, self.structure_fact_salt.ndim))

        self.structure_fact_charge = cp.roll(self.structure_fact_charge,
                [x//2 for x in self.structure_fact_charge.shape],
                axis=range(0, self.structure_fact_charge.ndim))
        self.struct_dists = wheres

        return

    def get_free_energy(self):
        #Function to get the overall free energy 
        #TODO: Figure out when we can avoid recomputing density 
        self.get_densities()
        self.update_normal_from_density()
        #initialize free energy array
        free_energy = cp.zeros_like(self.normal_w[0])

        #mu squared terms
        for i in range(self.normal_w.shape[0] - 1):
            free_energy += -(1 / (2 * self.normal_evalues[i])) \
                    * cp.square(self.normal_w[i]) 

        #mu linear terms
        
        for i in range(self.red_FH_mat.shape[0]-1): 
            for j in range(self.red_FH_mat.shape[0]-1): 
                free_energy += (self.normal_modes[j,i] * self.density_coeffs / \
                        self.normal_evalues[i]) * self.normal_w[i]

        free_energy -= self.normal_w[-1]
        #mu plus term
        #psi term
        grad_psi = self.gradient(self.psi)

        free_energy += self.tot_epsilon * cp.square(grad_psi) / 2
        
        total_free_energy = cp.sum(free_energy) * self.grid.dV

        #Partition energy contribution
        #TODO: Double check that this is exactly the same
        partition_energy = 0.
        for species in self.Q_dict:
            if species in self.poly_dict:
                partition_energy -= (self.poly_dict[species] \
                        / species.n_monomers) * cp.log(self.Q_dict[species] \
                        / self.poly_dict[species])
            elif species in self.solvent_dict:
                partition_energy -= self.solvent_dict[species] *\
                        cp.log(self.Q_dict[species] \
                        / self.solvent_dict[species])
            elif species in self.salts:
                salt_conc = self.salt_concs[self.salts.index(species)]
                partition_energy -=  salt_conc *\
                        cp.log(self.Q_dict[species] / salt_conc)
            else:
                print("Bad Species:", species)
                raise ValueError("Couldn't find species in any dictionary")
        total_free_energy += partition_energy * self.grid.V
        return total_free_energy

def s_step_2d(q_r, h, w_P, K2):
    #Take a single step along modified diffusion equation in 2d
    
    #e^(w(r,s) * h/2)
    exp_mult_comp= cp.ElementwiseKernel(
            'complex128 q_k, float64 K2, float64 h', 
            'complex128 out', 
            'out = q_k * exp(-K2 * h)',
            'exp_mult_comp')
    
    exp_mult= cp.ElementwiseKernel(
            'float64 q_r, float64 w_P, float64 h', 
            'float64 out', 
             'out = q_r * exp(-w_P * h/2)',
            'exp_mult')
    q_k = cp.zeros(q_r.shape, dtype=complex)
    q_r = exp_mult(q_r, w_P, h)
    #take e^gradient in fourier space
    q_k = cufft.fftn(q_r, s=q_r.shape)
    q_k = exp_mult_comp(q_k, K2, h)
    q_r = cufft.irfftn(q_k, s=q_k.shape)
    #e^(w(r,s) * h/2)
    q_r = exp_mult(q_r, w_P, h)
    return q_r

def integrate_s(struct, h_struct, species_dict, q_r_start, q_r_dag_start, K2):
    #integrates using each point in struct as an integration point
    #returns the q_r, q_r_dagger, and the key designating point along struct 
    #they belong too

    q_r = q_r_start
    q_r_dag = q_r_dag_start
    
    
    s_seg = len(struct)
    
    #write the list of all q_r points along polymer structure
    q_r_s = cp.zeros((s_seg+1, *q_r.shape))
    q_r_dag_s = cp.zeros((s_seg+1, *q_r.shape))
    
    #index to ensure sampling happened at the correct places
    i = 0
    
    #itialize q_r_s at q_r_start 
    q_r_s[0] = q_r
    q_r_dag_s[-1] = q_r_dag

    #advance, integrate and write key and q_r_s
    for bead in struct:
        q_r = s_step_2d(q_r, h_struct[i], species_dict[bead], K2)
        i+=1
        q_r_s[i] = q_r


    #retreat, integrate, and write key_dag and q_r_dag_s
    for bead in reversed(struct):
        i-=1
        q_r_dag = s_step_2d(q_r_dag, h_struct[i], species_dict[bead], K2)
        q_r_dag_s[i] = q_r_dag
    
    return q_r_s, q_r_dag_s
 


class CL_RK2(object):
    """
    docstring for complex langevin timestepping with RK2 integrator
    """

    def __init__(self, poly_sys, relax_rates, relax_temps, 
            psi_relax, psi_temp, full_epsilon=False):
        super(CL_RK2, self).__init__()

        self.ps = poly_sys
        if len({self.ps.w_all.shape[0], len(relax_rates),
            len(relax_temps)}) != 1:
            raise ValueError("Wrong sized relax rate or temperature")
        self.relax_rates = relax_rates
        self.temps = relax_temps
        self.psi_relax = psi_relax
        self.psi_temp = psi_temp
        self.full_epsilon = full_epsilon

    def d_H_d_mu(self, mu, phi_all):
        #derivative of H with respect to normal modes
        d_H = cp.zeros_like(mu)

        #declare a representation of phi_all that adds degenerate densities
        red_phi_all = self.ps.reduce_phi_all(phi_all)

        O_phi = cp.zeros_like(red_phi_all[:-1])
        O_phi = cp.reshape(
                self.ps.normal_modes.T@cp.reshape(red_phi_all[:-1],
                    (red_phi_all[:-1].shape[0],-1)), (O_phi.shape))


        #TODO: REMOVE THIS TESTING SMEARING 
        O_phi = self.ps.gaussian_smear(O_phi, self.ps.smear_const)

        #note transposes are just for reverse broadcasting
        d_H[:-1] = O_phi + ((-mu[:-1].T + \
            (self.ps.normal_modes.T@self.ps.density_coeffs)) \
                / self.ps.normal_evalues).T

        d_H[-1] = (cp.sum(phi_all, axis=0) - 1)
        return d_H

    def __full_RK2(self):
        #Concerted ETD2RK/RK for psi and mu all at once
        #WARNING: if this is called not as part of complex langevin it may have
        #unknown effects on underlying polymer structure
        self.ps.update_density_from_normal()
        self.ps.get_densities()

        w_all = cp.copy(self.ps.w_all)
        #get the phi densities and charge
#        total_charge_0 = self.ps.get_total_charge()
        
        mu = cp.copy(self.ps.normal_w)
        phi_all_0 = cp.copy(self.ps.phi_all)
        #Run first half of RK2 on mu    
        d_H = self.d_H_d_mu(mu, phi_all_0)
        
        d_mu1 = ((-self.relax_rates * cp.real(self.ps.gamma**2)) * d_H.T).T

        mu_hold = mu + d_mu1
        self.ps.normal_w = cp.copy(mu_hold)

        self.ps.update_density_from_normal()
        w_all_hold = cp.copy(self.ps.w_all)
    
        #First half of psi ETD2RK
#        h = self.psi_relax
       
        #TODO: Change to fourier space convolution

#        epsilon_k = cufft.fftn(self.ps.tot_epsilon/
#                cp.sum(self.ps.tot_epsilon), s=self.ps.tot_epsilon.shape)
#        #correct convolution to give c with variable epsilon (I think)
#        if self.full_epsilon: 
#            c = -(ndimage.convolve(self.ps.grid.k2, epsilon_k, mode='wrap',
#                origin=[-i//2 for i in epsilon_k.shape]) +
#                ndimage.convolve(self.ps.grid.k1, self.ps.grid.k1 
#                    * epsilon_k, mode ='wrap',
#                origin=[-i//2 for i in epsilon_k.shape]))
#        else:
#            c = -(self.ps.grid.k2) * cp.average(self.ps.tot_epsilon)# +\
#                cp.fft.fftn(grad_psi * grad_epsilon, s=self.ps.psi.shape))

        #This will make it stop complaining about zero division, but you MUST 
        #correct the first term later on
#        c.flat[0] = 1
#        psi_k = cufft.fftn(self.ps.psi, s=self.ps.psi.shape) 
    
#        F_n = -cufft.fftn(total_charge_0, s=total_charge_0.shape)
    
    
#        a_n = psi_k * cp.exp(c*h) + F_n * (cp.exp(c*h) - 1)/c
        #correcting the first term 
#        a_n.flat[0] = psi_k.flat[0] + F_n.flat[0] * h 
        
#        psi_hold = cufft.irfftn(a_n, s=self.ps.psi.shape)
#        self.ps.psi = psi_hold
        #finished both first halves, have intermediate species
         
        #Get phis associated with intermediate species
        self.ps.get_densities()
        phi_all_1 = cp.copy(self.ps.phi_all)
#        total_charge_1 = self.ps.get_total_charge()
    
        #second half of RK2 for mu
        d_H2 = self.d_H_d_mu(mu_hold, phi_all_1)
    
        d_mu2 = ((-self.relax_rates * cp.real(self.ps.gamma**2)) * d_H2.T).T
        new_mu = mu + d_mu2
        new_mu[-1] -= cp.average(new_mu[-1])
        
        #Second half of ETD2RK for psi
#        epsilon_k = cufft.fftn(self.ps.tot_epsilon/
#                cp.sum(self.ps.tot_epsilon), s=self.ps.tot_epsilon.shape)
        #correct convolution to give c with variable epsilon (I think)
#        if self.full_epsilon: 
#            c = -(ndimage.convolve(self.ps.grid.k2, epsilon_k, mode='wrap',
#                origin=[-i//2 for i in epsilon_k.shape]) +
#                ndimage.convolve(self.ps.grid.k1, self.ps.grid.k1 
#                    * epsilon_k, mode ='wrap',
#                origin=[-i//2 for i in epsilon_k.shape]))
#        else:
#            c = -(self.ps.grid.k2) * cp.average(self.ps.tot_epsilon)# +\
#                cp.fft.fftn(grad_psi * grad_epsilon, s=self.ps.psi.shape))

        #Again, MUST be corrected later
#        c.flat[0] = 1

#        F_an =  -cufft.fftn(total_charge_1, s=total_charge_1.shape)
    
#        u_n1 = a_n + (F_an - F_n) * (cp.exp(c*h)-1-h*c)/(h*cp.square(c))
        
        #again correcting the first term
#        u_n1.flat[0]= a_n.flat[0] + (F_an.flat[0] - F_n.flat[0]) * (h/2)
#        new_psi = cufft.irfftn(u_n1, s=self.ps.psi.shape)
        cp.cuda.stream.get_current_stream().synchronize()
        return new_mu

    def complex_langevin(self):
        #full complex langevin time step 
        new_mu = self.__full_RK2()
        noise = cp.zeros_like(self.ps.normal_w)
    
        #add in noise to mu
        # do not add noise to purely imaginary fields 
        # different from duchs
        d_i = cp.append(self.ps.normal_evalues, 1)
        adjusted_temps = self.temps * (d_i < 0)
#        adjusted_temps = self.temps 
        for i in range(self.relax_rates.shape[0]):
            noise_hold = cp.random.normal(0, cp.sqrt(2 * self.relax_rates[i]\
                    * adjusted_temps[i] / self.ps.grid.dV), 
                    size = self.ps.normal_w[i].size)
            noise[i] = cp.reshape(noise_hold, noise[i].shape)
    
        new_mu += noise
    
        #add in noise to psi
#        psi_noise = cp.zeros_like(self.ps.psi)

#        psi_noise = cp.random.normal(0, cp.sqrt(2 * self.psi_relax \
#                * self.psi_temp / self.ps.grid.dV), size = self.ps.psi.size) 
#        psi_noise = cp.reshape(psi_noise, self.ps.psi.shape)


#        new_psi += psi_noise
        self.ps.normal_w = new_mu


#        self.ps.psi = new_psi

        self.ps.set_field_averages()
        self.ps.update_density_from_normal() 

        cp.cuda.stream.get_current_stream().synchronize()
        return 
