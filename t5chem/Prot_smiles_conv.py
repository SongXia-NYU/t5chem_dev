from rdkit.Chem.rdmolfiles import MolFromFASTA,MolToSmiles
from numpy import random

def AAtoSmiles(sequence):
	mol_objects = []
	for aa in sequence:
		mol_objects.append(MolFromFASTA(aa))
	smiles_string = ""
	for molecule in mol_objects:
		smiles_string += "<mod>"
		smiles_string += MolToSmiles(molecule)
		smiles_string += "</mod>"
	return smiles_string


def pick_random_partition(line):
	start = random.randint(0, len(line))
	end = random.randint(start, start + 5)
	if end >= len(line):
		end = len(line) - 1
	if start == end:
		return pick_random_partition(line)
	return start, end

def write_mixed_file(origin_file,new_file, chance=0.25):
	remove_org = False
	file_mix_tag = open(new_file, "w")
	lines = [line.strip() for line in open(origin_file)]
	for line in lines:
		modified_line = None
		if random.uniform(0, 1) <= chance:
			start,end = pick_random_partition(line)
			smiles = AAtoSmiles(line[start:end])
			if "<mod>" + smiles + "</mod>" == "<mod></mod>":
				print(start, end)
			#smiles = "<mod>" + smiles + "</mod>"
			left_partition = line[0:start]
			right_partition = line[end:len(line)]
			modified_line = left_partition + smiles + right_partition
		if remove_org:
			if modified_line:
				file_mix_tag.write(modified_line + "\n")
			else:
				file_mix_tag.write(line + "\n")
		else:
			if modified_line:
				file_mix_tag.write(modified_line + "\n")
			file_mix_tag.write(line + "\n")

	print("finished")
if __name__ == "__main__":
	origin_file = "/scratch/tk2801/t5chem_dev_song/t5chem_dev/data_molprot/val_prot.txt"
	new_file = "/scratch/tk2801/t5chem_dev_song/t5chem_dev/data_molprot/val_mixed_prot.txt"
	write_mixed_file(origin_file,new_file)
