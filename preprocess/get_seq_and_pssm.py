import os, subprocess, argparse
from multiprocessing import Pool
import numpy as np


BLAST_DB = 'your_database_path/nrdb90/nrdb90'
BLAST = './lib/psiblast'

''' blosum init start'''
blosum62 = {}
blosum_reader = open("./lib/blosum62", 'r')
count = 0
for line in blosum_reader:
    count = count + 1
    if count <= 7:
        continue
    line = line.strip('\r').split()
    blosum62[line[0]] = [float(x) for x in line[1:21]]
''' blosum init end'''

def get_protein_blosum62(fasta_file):
    # get sequence
    with open(fasta_file, 'r') as f:
        seq = f.readlines()[1].strip()
    protein_lst = []
    for aa in seq:
        aa = aa.upper()
        protein_lst.append(blosum62[aa])
    matrix = np.asarray(protein_lst)

    return matrix


def getFasta(dsspfile):
    with open(dsspfile, 'r') as f:
        text = f.readlines()
    fasta = ''
    process_flag = False
    try:
        for line in text:
            tmp = line.strip()
            if tmp[0] == '#':
                process_flag = True
                continue

            if process_flag:
                residue_id = line[7:10].strip()
                if residue_id == '':  
                    continue
                AA = line[13]
                fasta += AA
    except:
        print(dsspfile)

    return fasta

def get_name_list(path):
    prots = []
    files = os.listdir(path)
    for file in files[:]:
        prot_name = os.path.splitext(file)[0]
        
        prots.append(prot_name)
    return prots

no_pssm_list = []

PSSM_Outputpath = ''
error_filename = 'no_pssm_fasta.txt'
def main(inputpath, outputpath, name_list, processors=28):
    global BLAST
    global BLAST_DB
    global PSSM_Outputpath

    PSSM_Outputpath = os.path.join(outputpath, 'pssm')
    error_file = os.path.join(os.path.dirname(PSSM_Outputpath), error_filename)
    if os.path.exists(error_file):
        os.remove(error_file)
    Fasta_Outputpath = os.path.join(outputpath, 'fasta')
    dssppath = inputpath
    if not os.path.exists(PSSM_Outputpath):
        os.makedirs(PSSM_Outputpath)
    if not os.path.exists(Fasta_Outputpath):
        os.makedirs(Fasta_Outputpath)
    if name_list != '':
        with open(name_list, 'r') as f:
            prots = f.read().strip().split()
    else:
        prots = get_name_list(dssppath)

    print('start parsing', len(prots))

    pool = Pool(processors)
    pool.map(process_file, [(os.path.join(PSSM_Outputpath, prot+'.pssm'),
                                    os.path.join(Fasta_Outputpath, prot+'.fasta'),
                                    os.path.join(dssppath, prot+'.dssp'))
                                    for prot in prots]) 
    pool.close()
    pool.join()

    return PSSM_Outputpath

def pool_generate_pssm(arg):
    generate_fasta_and_pssm(arg[0], arg[1], arg[2])

def get_fasta(dssp_file, fasta_file):
    prot_name = os.path.splitext(dssp_file.split('/')[-1])[0]
    fastatext = '>' + prot_name + '\n' + getFasta(dssp_file)
    with open(fasta_file, 'w') as f:
        f.writelines(fastatext)

def generate_fasta_and_pssm(dssp_file, fasta_file, pssm_file):
    get_fasta(dssp_file, fasta_file)
    run_generate_pssm(pssm_file, fasta_file)

def run_generate_pssm(pssm_file, fasta_file):
    global BLAST
    global BLAST_DB
    outfmt_type = 5
    num_iter = 10
    evalue_threshold = 0.01

    cmd = ' '.join([BLAST,
                    '-query ' + fasta_file,
                    '-db ' + BLAST_DB,
                    '-out ' + '/dev/null', 
                    '-evalue ' + str(evalue_threshold),
                    '-num_iterations ' + str(num_iter),
                    '-outfmt ' + str(outfmt_type),
                    '-out_ascii_pssm ' + pssm_file,  
                    '-num_threads ' + '8']
                    )
    # print(cmd)
    info = subprocess.call(cmd, shell=True)
        
    if not os.path.isfile(pssm_file): # replace with a blosum 
        prot_name = os.path.splitext(fasta_file.split('/')[-1])[0]
        matrix = get_protein_blosum62(fasta_file) 
        npy_path = os.path.join(PSSM_Outputpath, prot_name + '.npy')
        np.savetxt(npy_path, matrix)       
        if not os.path.isfile(os.path.join(PSSM_Outputpath, prot_name + '.npy')):
            print('no pssm or blomsum', fasta_file)
        with open(os.path.join(os.path.dirname(PSSM_Outputpath), error_filename), 'a') as ff:
            ff.writelines(fasta_file + '\n')

# 包装函数：适配多进程输入
def process_file(file_pair):
    pssm_file, fasta_file, dssp_file = file_pair
    generate_fasta_and_pssm(dssp_file, fasta_file, pssm_file)
    return dssp_file  

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inputpath', '-i', type=str, default="", required=True, help="a path of input dssp files")
    parser.add_argument('--list', '-l', type=str, default="", required=False, help="a protname list")
    parser.add_argument('--processors', '-p', type=int, default=8, required=False)
    parser.add_argument('--outputpath', '-o', type=str, default="", required=False,
                        help="a path for saving related output: dir`fasta` and `pssm` will be created")
    return parser.parse_args()

    

def script():
    args = arg_parse()
    input_path = args.inputpath
    output_path = args.outputpath if args.outputpath != "" else os.path.join(input_path, 'output')
    main(input_path, output_path, name_list=args.list, processors=args.processors)

if __name__=='__main__':
    script()
    
