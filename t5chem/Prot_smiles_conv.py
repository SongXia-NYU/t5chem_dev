from rdkit.Chem.rdmolfiles import MolFromFASTA,MolToSmiles
from numpy import random

def AAtoSmiles(sequence):
	mol_obj = MolFromFASTA(sequence)
	return MolToSmiles(mol_obj)
def pick_random_partition():
	start = random.randint(0, len(line))
	end = random.randint(start, start + 5)
	if end >= len(line):
		end = len(line) - 1
	if start == end:
		return pick_random_partition()
	return start, end

chance = 0.25
file_mix_tag = open("val_mix_tagged.txt", "w")
lines = [line.strip() for line in open("val_prot.txt")]
for line in lines:
	modified_line = None
	if random.uniform(0, 1) <= chance:
		start,end = pick_random_partition()
		smiles = AAtoSmiles(line[start:end])
		if "<chem>" + smiles + "</chem>" == "<chem></chem>":
			print(start, end)
		smiles = "<chem>" + smiles + "</chem>"
		left_partition = line[0:start]
		right_partition = line[end:len(line)]
		modified_line = left_partition + smiles + right_partition
	if modified_line:
		file_mix_tag.write(modified_line + "\n")
	else:
		file_mix_tag.write(line + "\n")

print("finished")