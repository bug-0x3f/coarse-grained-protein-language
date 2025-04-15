from multiprocessing import Pool
import os, time
import subprocess
import tempfile, shutil

timestamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
OUTPUT_PATH = ''
error_file = None
def multi_generate_dssp(arg):
    pdb_file_path, dssp_file_path = arg
    if os.path.exists(dssp_file_path): # 跳过已经生成的
        return
    command = ["dssp", "-i", pdb_file_path, "-o", dssp_file_path]
    global OUTPUT_PATH
    try:
        res = subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:

        with open(error_file, 'a') as f:
            f.write(pdb_file_path + '\n')
        print(str(e.stderr.decode('utf-8')))
        
        global error_dssp
        print(f"生成 {dssp_file_path} 时出错: {e}")

def get_seq_from_pdb(pid, pdb_file):
    AA_dic = {'GLY':'G','ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','TYR':'Y','ASP':'D','ASN':'N',
          'GLU':'E','LYS':'K','GLN':'Q','MET':'M','SER':'S','THR':'T','CYS':'C','PRO':'P','HIS':'H','ARG':'R'}
    AA = ['GLY','ALA','VAL','LEU','ILE','PHE','TRP','TYR','ASP','ASN',
        'GLU','LYS','GLN','MET','SER','THR','CYS','PRO','HIS','ARG']
    DA = ['DA', 'DC', 'DT', 'DG']
    RA = ['A', 'C', 'T', 'U', 'G']

    from Bio.PDB.PDBParser import PDBParser
    import warnings
    warnings.filterwarnings("ignore")

    parser = PDBParser()
    try:
        structure = parser.get_structure(pid, pdb_file)
    except Exception as e:
        raise ValueError(f"fail to get dssp: {pdb_file}")
        
    
    atom_list = []
    ids = []
    residues = []
    for model in structure:
        n = 0
        for chain in model:
            if n == 0:
                for residue in chain:
                    if str(residue.get_resname()).strip() in AA:
                        ids.append(residue.get_id()[1])
                        residues.append(AA_dic[residue.get_resname()])
                        for atom in residue:
                            atom_list.append(atom)
            else:
                for residue in chain:
                    if (str(residue.get_resname()).strip() in DA) or (str(residue.get_resname()).strip() in RA):
                        for atom in residue:
                            atom_list.append(atom)
                    else:
                        pass
            n += 1

    seq = ''.join(residues)
    return seq


def main(input_path, output_path, name_list=[]):

    if not os.path.exists(input_path):
        raise FileNotFoundError("filepath not found" + input_path)
    os.makedirs(output_path, exist_ok=True)

    print("-----generate dssp start-----")

    global OUTPUT_PATH
    OUTPUT_PATH = output_path
    global error_file
    if OUTPUT_PATH[-1] != '/':
        error_file = os.path.join(os.path.dirname(OUTPUT_PATH), f'no_dssp_pdb_{timestamp}.txt')
    else:
        error_file = os.path.join(os.path.dirname(os.path.dirname(OUTPUT_PATH)), f'no_dssp_pdb_{timestamp}.txt')
    
    pdb_list = []
    dssp_list = []
    if name_list == []:
        for file in os.listdir(input_path):
            prot_name = file.split('.')[0]
            if file.endswith(".pdb") or file.endswith(".ent"):
                pdb_file_path = os.path.join(input_path, file)
                dssp_file_path = os.path.join(output_path, os.path.splitext(file)[0] + ".dssp")
                pdb_list.append(pdb_file_path)
                dssp_list.append(dssp_file_path)
    else:
        for prot_name in name_list:
            pdb_file_path = os.path.join(input_path, prot_name +'.pdb')
            if not os.path.exists(pdb_file_path):
                pdb_file_path = os.path.join(input_path, prot_name +'.ent')
            dssp_file_path = os.path.join(output_path, prot_name + ".dssp")
            pdb_list.append(pdb_file_path)
            dssp_list.append(dssp_file_path)

    pool = Pool(32)
    params = [arg for arg in zip(pdb_list, dssp_list)]
    pool.map(multi_generate_dssp, params)
    pool.close()
    pool.join()


    print("-----generate dssp done-----")
    print(error_file)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', '-d', type=str, required=True) 
    parser.add_argument('--name_list', '-l', type=str, default='', required=False)
    
    parser.add_argument('--output_path', '-o', type=str, 
                        required=True, help='需以dssp结尾')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    input_path = args.pdb_dir
    output_path = args.output_path
    prots = []
    if args.name_list != '':
        with open(args.name_list, 'r') as f:
            prots = f.read().strip().split()
    main(input_path, output_path, name_list=prots)
