from rdkit.Chem.rdmolfiles import MolFromFASTA,MolToSmiles
from numpy import random

def AAtoSmiles(sequence):
	mol_obj = MolFromFASTA(sequence)
	return MolToSmiles(mol_obj)
def pick_random_partition(line):
	start = random.randint(0, len(line))
	end = random.randint(start, start + 5)
	if end >= len(line):
		end = len(line) - 1
	if start == end:
		return pick_random_partition()
	return start, end

def write_mixed_file(origin_file,new_file, chance =0.25):
	file_mix_tag = open(new_file, "w")
	lines = [line.strip() for line in open(origin_file)]
	for line in lines:
		modified_line = None
		if random.uniform(0, 1) <= chance:
			start,end = pick_random_partition(line)
			smiles = AAtoSmiles(line[start:end])
			if "<mod>" + smiles + "</mod>" == "<mod></mod>":
				print(start, end)
			smiles = "<mod>" + smiles + "</mod>"
			left_partition = line[0:start]
			right_partition = line[end:len(line)]
			modified_line = left_partition + smiles + right_partition
		if modified_line:
			file_mix_tag.write(modified_line + "\n")
		else:
			file_mix_tag.write(line + "\n")

	print("finished")
if __name__ == "__main__":
	origin_file = "train_prot.txt"
	new_file = "train_mixed_prot.txt"
	write_mixed_file(origin_file,new_file)
