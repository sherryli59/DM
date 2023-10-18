from ovito.io import import_file
from ovito import modifiers
from ovito.vis import Viewport, OSPRayRenderer
import numpy as np
import math

def assign_particle_radii(frame, data):
    atom_types = data.particles_.particle_types_
    atom_types.type_by_id_(1).radius = 0.5   # Fe atomic radius assigned to atom type 1
    atom_types.type_by_id_(1).color =  (0.92, 0.4 ,0.3 )

def make_movie(data_path,filename):
    pipeline = import_file(data_path+filename)
    pipeline.add_to_scene()
    data = pipeline.compute()
    pipeline.modifiers.append(assign_particle_radii)

    cell_vis = pipeline.source.data.cell.vis
    cell_vis.enabled = False

    vp = Viewport(type = Viewport.Type.Ortho, camera_dir = (2, 1, -1))
    vp.camera_pos = (-10, -15, 15)
    vp.camera_dir = (2, 3, -3)
    vp.fov = math.radians(60.0)
    vp.zoom_all()
    vp.render_image(filename=data_path+'traj.png', 
                    size=(1080,960), 
                    #fps=50,
                    frame = 499)
                    #range = (0,10),
                    #renderer=OSPRayRenderer(max_ray_recursion=3))

if __name__ == "__main__":
    data_path = "/mnt/ssd/ml-boilerplate/logs/runs/LJ/vp_vf/"
    filename = "traj.npy"
    make_movie(data_path,filename)