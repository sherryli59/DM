from math import pi
import os
from torch import sin, cos
from torch.linalg import norm
import torch
import numpy as np
import mdtraj as md

# Suppress scientific notation printouts and change default precision
torch.set_printoptions(precision=4, sci_mode=False)


class AtomCoords():

    def __init__(self, traj=None, permute=None, device="cuda:0"):
        self.device = device
        self.traj = traj
        self.permute = permute
        # Internal Coordinate Connectivity
        self.connectivity = None
        self.angleconnectivity = None
        self.dihedralconnectivity = None
        # Internal Coordinates
        self.distances = None
        self.angles = None
        self.dihedrals = None
        if traj is not None:
            self.natoms = traj.topology.n_atoms
            self.atomnos = torch.tensor([atom.element.number for atom in traj.topology.atoms])
            self.all_xyz = torch.tensor(traj.xyz).to(device)
            if permute is not None:
                self.all_xyz = self.all_xyz[:,permute]
                self.atomnos = self.atomnos[permute]
                self.inv_permute = torch.argsort(torch.torch.tensor(permute))
            self.xyz = self.all_xyz.clone()
            self.nframes = self.xyz.shape[0]
            self._build_zmatrix()
        self.distancematrix = None



    def _build_distance_matrix(self):
        """Build distance matrix between all atoms
        """
        coords = self.xyz
        pair_vec = (coords.unsqueeze(-2) - coords.unsqueeze(-3))
        distances = norm(pair_vec.float(), axis=-1)
        self.distancematrix = distances

    def _build_zmatrix(self):
        """
       'Z-Matrix Algorithm'
        Build main components of zmatrix:
        Connectivity vector
        Distances between connected atoms (atom >= 1)
        Angles between connected atoms (atom >= 2)
        Dihedral angles between connected atoms (atom >= 3)
        """
        self._build_distance_matrix()
        self.nframes = self.distancematrix.shape[0]
        # self.connectivity[i] tells you the index of 2nd atom connected to atom i
        self.connectivity = torch.zeros(self.natoms,dtype=int).to(self.device)

        # self.angleconnectivity[i] tells you the index of
        #    3rd atom connected to atom i and atom self.connectivity[i]
        self.angleconnectivity = self.connectivity.clone()

        # self.dihedralconnectivity tells you the index of 4th atom connected to
        #    atom i, atom self.connectivity[i], and atom self.angleconnectivity[i]
        self.dihedralconnectivity = self.connectivity.clone()
        # Starts with r1
        self.distances = torch.zeros(self.nframes,self.natoms).to(self.device)
        # Starts with a2
        self.angles = self.distances.clone()
        # Starts with d3
        self.dihedrals = self.distances.clone()

        atoms = range(1, self.natoms)
        for atom in atoms:
            # For current atom, find the nearest atom among previous atoms
            distvector = self.distancematrix[:,atom,:atom]
            distmin, nearestatom = torch.min(distvector,axis=-1)
            self.connectivity[atom] = nearestatom[0]
            self.distances[:,atom] = distvector[:,nearestatom[0]]
            if norm(distmin- self.distances[:,atom])>0:
                print("WARNING: Nearest atom nonunique or not consistent across frames for atom {}".format(atom))

            # Compute Angles
            if atom >= 2:
                atms = [0, 0, 0]
                atms[0] = atom
                atms[1] = self.connectivity[atms[0]]
                atms[2] = self.connectivity[atms[1]]
                if atms[2] == atms[1]:
                    for idx in range(1, len(self.connectivity[:atom])):
                        if self.connectivity[idx] in atms and not idx in atms:
                            atms[2] = idx
                            break

                self.angleconnectivity[atom] = atms[2]

                self.angles[:,atom] = self._calc_angle(atms[0], atms[1], atms[2])

            # Compute Dihedral Angles
            if atom >= 3:
                atms = [0, 0, 0, 0]
                atms[0] = atom
                atms[1] = self.connectivity[atms[0]]
                atms[2] = self.angleconnectivity[atms[0]]
                atms[3] = self.angleconnectivity[atms[1]]
                if atms[3] in atms[:3]:
                    for idx in range(1, len(self.connectivity[:atom])):
                        if self.connectivity[idx] in atms and not idx in atms:
                            atms[3] = idx
                            break

                self.dihedrals[:,atom] =\
                    self._calc_dihedral(atms[0], atms[1], atms[2], atms[3])
                if torch.any(self.dihedrals[:,atom].isnan()):
                    # TODO: Find explicit way to denote undefined dihedrals
                    self.dihedrals[:, atom] = torch.zeros(self.nframes)

                self.dihedralconnectivity[atom] = atms[3]

    def _calc_angle(self, atom1, atom2, atom3):
        """Calculate angle between 3 atoms"""
        coords = self.xyz
        vec1 = coords[:,atom2] - coords[:,atom1]
        uvec1 = vec1 / norm(vec1, axis=-1).unsqueeze(-1)
        vec2 = coords[:,atom2] - coords[:,atom3]
        uvec2 = vec2 / norm(vec2, axis=-1).unsqueeze(-1)
        return torch.arccos(torch.sum(uvec1* uvec2, dim=-1))

    def _calc_dihedral(self, atom1, atom2, atom3, atom4):
        """
           Calculate dihedral angle between 4 atoms
           For more information, see:
               http://math.stackexchange.com/a/47084
        """
        coords = self.xyz
        # Vectors between 4 atoms
        b1 = coords[:,atom2] - coords[:,atom1]
        b2 = coords[:,atom2] - coords[:,atom3]
        b3 = coords[:,atom4] - coords[:,atom3]

        # Normal vector of plane containing b1,b2
        n1 = torch.cross(b1, b2, dim=-1)
        un1 = n1 / norm(n1, axis=-1).unsqueeze(-1)

        # Normal vector of plane containing b1,b2
        n2 = torch.cross(b2, b3, dim=-1)
        un2 = n2 / norm(n2, axis=-1).unsqueeze(-1)

        # un1, ub2, and m1 form orthonormal frame
        ub2 = b2 / norm(b2, axis=-1).unsqueeze(-1)
        um1 = torch.cross(un1, ub2, dim=-1)

        # dot(ub2, n2) is always zero
        x = torch.sum(un1* un2, dim=-1)
        y = torch.sum(um1* un2, dim=-1)

        dihedral = torch.arctan2(y, x)
        dihedral = dihedral + (dihedral < 0) * 2 * pi
        return dihedral


# Above are utility functions for converting xyz to zmat,
# Below are utility functions for converting zmat to xyz



    def _build_xyz(self):
        """ Build xyz representation from z-matrix"""
        self.xyz = torch.zeros(self.nframes,self.natoms,3).to(self.device)
        for i in range(self.natoms):
            self.xyz[:,i] = self._calc_position(i)

    def _calc_position(self, i):
        """Calculate position of another atom based on internal coordinates"""

        if i > 1:
            j = self.connectivity[i]
            k = self.angleconnectivity[i]
            l = self.dihedralconnectivity[i]

            # Prevent doubles
            if k == l and i > 0:
                for idx in range(1, len(self.connectivity[:i])):
                    if self.connectivity[idx] in [i, j, k] and not idx in [i, j, k]:
                        l = idx
                        break

            avec = self.xyz[:,j]
            bvec = self.xyz[:,k]

            dst = self.distances[:,i]
            ang = self.angles[:,i]

            if i == 2:
                # Third atom will be in same plane as first two
                tor =  torch.tensor(pi / 2).to(self.device)
                cvec = torch.tensor([0, 1, 0]).to(self.device)
            else:
                # Fourth + atoms require dihedral (torsional) angle
                tor = self.dihedrals[:, i]
                cvec = self.xyz[:, l]

            v1 = avec - bvec
            v2 = avec - cvec

            n = torch.cross(v1, v2, dim=-1)
            nn = torch.cross(v1, n, dim=-1)

            n /= norm(n, axis=-1).unsqueeze(-1)
            nn /= norm(nn, axis=-1).unsqueeze(-1)
            n *= -sin(tor).unsqueeze(-1)
            nn *= cos(tor).unsqueeze(-1)

            v3 = n + nn
            v3 /= norm(v3, axis=-1).unsqueeze(-1)
            v3 *= dst.unsqueeze(-1) * sin(ang).unsqueeze(-1)

            v1 /= norm(v1, axis=-1).unsqueeze(-1)
            v1 *= dst.unsqueeze(-1) * cos(ang).unsqueeze(-1)

            position = avec + v3 - v1

        elif i == 1:
            # Second atom dst away from origin along Z-axis
            j = self.connectivity[i]
            dst = self.distances[:,i]
            position = torch.stack([self.xyz[:,j,0] + dst, self.xyz[:,j,1],
                                     self.xyz[:,j, 2]],axis=-1)

        elif i == 0:
            # First atom at the origin
            position = torch.zeros((len(self.xyz), 3)).to(self.device)

        return position

    def set_xyz(self,xyz):
        self.xyz = xyz.to(self.device)
        self.nframes = xyz.shape[0]
        self.natoms = xyz.shape[1]

    def get_zmat(self, xyz=None, need_permute=True, batchsize=10000):
        if xyz is not None:
            if xyz.dim() == 2:
                xyz = xyz.unsqueeze(0)
            if need_permute and self.permute is not None:
                xyz = xyz[:,self.permute]
            self.all_xyz = xyz.to(self.device)
        assert self.all_xyz is not None, "xyz coordinates not set"
        indices= torch.arange(len(self.all_xyz))[::batchsize]
        indices = torch.cat([indices, torch.tensor([len(self.all_xyz)])]).int()
        self.all_distances = torch.empty(0, self.all_xyz.shape[1]).to(self.device)
        self.all_angles = self.all_distances.clone()
        self.all_dihedrals = self.all_distances.clone()
        for i in range(len(indices)-1):
            self.set_xyz(self.all_xyz[indices[i]:indices[i+1]])
            self._build_zmatrix()
            self.all_distances = torch.cat((self.all_distances, self.distances),axis=0)
            self.all_angles = torch.cat((self.all_angles, self.angles),axis=0)
            self.all_dihedrals = torch.cat((self.all_dihedrals, self.dihedrals),axis=0)
        return torch.stack((self.all_distances, self.all_angles, self.all_dihedrals),dim=1)
    
    def set_zmat(self, distances, angles, dihedrals):
        if distances.dim() == 1:
            distances = distances.unsqueeze(0)
        if angles.dim() == 1:
            angles = angles.unsqueeze(0)
        if dihedrals.dim() == 1:
            dihedrals = dihedrals.unsqueeze(0)
        self.distances = distances.to(self.device)
        self.angles = angles.to(self.device)
        self.dihedrals = dihedrals.to(self.device)
        self.natoms = distances.shape[1]
        self.nframes = distances.shape[0]

    
    def get_xyz(self,restore_order=True, batchsize=10000, **kwargs):
        distances = kwargs.get("distances", None)
        angles = kwargs.get("angles", None)
        dihedrals = kwargs.get("dihedrals", None)
        if distances is not None and angles is not None and dihedrals is not None:
            self.all_distances = distances
            self.all_angles = angles
            self.all_dihedrals = dihedrals
        else: 
            zmat = kwargs.get("zmat", None)
            if zmat is not None:
                self.all_distances = zmat[:,0]
                self.all_angles = zmat[:,1]
                self.all_dihedrals = zmat[:,2]

        assert self.all_distances is not None and self.all_angles is not None and self.all_dihedrals is not None, "Need to provide distances, angles, dihedrals"
        indices= torch.arange(len(self.all_distances))[::batchsize]
        indices = torch.cat([indices, torch.tensor([len(self.all_distances)])]).int()
        self.all_xyz = torch.empty(0, self.all_distances.shape[1], 3).to(self.device)
        for i in range(len(indices)-1):
            self.set_zmat(self.all_distances[indices[i]:indices[i+1]],
                           self.all_angles[indices[i]:indices[i+1]], self.all_dihedrals[indices[i]:indices[i+1]])
            self._build_xyz()
            self.all_xyz = torch.cat((self.all_xyz, self.xyz),axis=0)
        if restore_order:
            if self.permute is not None:
                self.all_xyz = self.all_xyz[:,self.inv_permute]
        return self.all_xyz



class ZmatConversion():
    def __init__(self, traj=None, file=None,permute=None):
        if traj is not None:
            self.traj = traj
        elif file is not None:
            ext = os.path.splitext(file)[-1].lower()
            if ext == '.h5':
                self.traj = md.load_hdf5(file)
            elif ext == '.pdb':
                self.traj = md.load_pdb(file)
            else:
                raise ValueError("Unsupported file format")
        else:
            self.traj = None #if no file is provided, need to call xyz_to_zmat first to set connectivities
        self.sys = AtomCoords(self.traj,permute=permute)

    def xyz_to_zmat(self, xyz, need_permute=True, flatten=True, group_by_atom=True):
        zmat = self.sys.get_zmat(xyz, need_permute=need_permute)
        if flatten:
            zmat = self.flatten(zmat, group_by_atom=group_by_atom)
        return zmat
    
    def encode(self, x):
        return self.xyz_to_zmat(x)
    
    def zmat_to_xyz(self, zmat, flattened=True, restore_order=True, null_val=0):
        if flattened:
            zmat = self.parse(zmat, null_val=null_val)
        xyz = self.sys.get_xyz(zmat=zmat,restore_order=restore_order)
        return xyz

    def decode(self, z):
        return self.zmat_to_xyz(z)
    
    @staticmethod
    def parse(zmat, null_val=0, grouped_by_atom=True):
        #flattened data (zeros ommitted) -> (distances,angles,dihedrals)
        natoms = zmat.shape[1]//3 + 2
        if zmat.dim() == 1:
            zmat = zmat.unsqueeze(0)
        if grouped_by_atom: 
            # zmat is in the form of [b,b,a,b,a,d,...]
            # need to insert columns at 0,1,2,4,5,8
            col_3 = torch.full((zmat.shape[0],3),null_val).to(zmat.device)
            col_2 = torch.full((zmat.shape[0],2),null_val).to(zmat.device)
            col_1  = torch.full((zmat.shape[0],1),null_val).to(zmat.device)
            zmat = torch.cat((col_3,zmat[:,0].unsqueeze(1),col_2,zmat[:,1:3],col_1,zmat[:,3:]),axis=1)
            zmat = zmat.reshape(zmat.shape[0],-1,3).permute(0,2,1)
            return zmat
        else:
            natoms = zmat.shape[1]//3 + 2
            nframes = zmat.shape[0]
            distances = zmat[:,:natoms-1]
            distances = torch.cat((torch.full((nframes,1),null_val).to(zmat.device), distances),axis=1)
            angles = zmat[:,natoms-1:2*natoms-3]
            angles = torch.cat((torch.full((nframes,2),null_val).to(zmat.device), angles),axis=1)
            dihedrals = zmat[:,2*natoms-3:]
            dihedrals = torch.cat((torch.full((nframes,3),null_val).to(zmat.device), dihedrals),axis=1)
            return torch.stack((distances,angles,dihedrals),dim=1)

    @staticmethod
    def flatten(data,group_by_atom=True):
        # (distances,angles,dihedrals) -> flattened data (zeros ommitted)
        if group_by_atom:
            data = data.permute(0,2,1)
            data = data.reshape(data.shape[0],-1)
            index = np.array(range(data.shape[1]))
            del_index = np.array([0,1,2,4,5,8]) #redundant zeros
            new_index = np.delete(index,del_index)
            flattened_zmat = data[:,new_index]
        else:
            distances = data[:,0,1:]
            angles = data[:,1,2:]
            dihedrals = data[:,2,3:]
            flattened_zmat = torch.cat((distances,angles,dihedrals),dim=1)
        return flattened_zmat
