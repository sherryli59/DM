import argparse
from psfgen import PsfGen

parser = argparse.ArgumentParser()
parser.add_argument("--lipid_type", type=str, default="dlpc",
                    choices=["dlpc", "dppc", "dmpc", "dopc", "pope"])

config = parser.parse_args()
lipid_type = config.lipid_type
base_filename = lipid_type
lipids = {lipid_type: 1.0}

temperature_pairs = dict(dlpc=303,
                         dppc=323.15,
                         dopc=296,
                         dmpc=303,
                         popc=303, pope=310.15)
integrator_temp = temperature_pairs[lipid_type]

gen = PsfGen(output="/dev/null")  # Supresses all output

gen.read_topology("charmm_files/par_all36_cgenff.prm")
gen.read_topology("charmm_files/top_all36_cgenff.rtf")
gen.read_topology(
    "charmm_files/toppar_all36_lipid_cholesterol.str")
gen.read_topology("charmm_files/par_all36_lipid.prm")
gen.read_topology("charmm_files/top_all36_lipid.rtf")


lipid_filename = ("lipid_pdb/" + lipid_type + "_" + str(1) + "_" + str(1) + ".pdb")
gen.add_segment(segid="CM", pdbfile=lipid_filename)
gen.read_coords(segid="CM", filename=lipid_filename)
gen.write_psf(filename="input/"+base_filename + ".psf")
gen.write_pdb(filename ="input/"+ base_filename + ".pdb")

