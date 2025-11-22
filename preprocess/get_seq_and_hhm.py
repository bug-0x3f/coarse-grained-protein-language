
import time, os, subprocess, argparse
from multiprocessing import Pool

HHBLITS = './lib/hhblits'
HHBLITS_DB = '/your_database_path/uniprot20_2016_02/uniprot20_2016_02'
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
                if residue_id == '':  # 跳过注释行
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
        # if prot_name.count('.') > 0:
        #     print(prot_name)
        prots.append(prot_name)
    return prots

no_pssm_list = []


HHM_Outputpath = ''
error_filename = ''
def main(inputpath, outputpath, name_list, processers=8):
    global HHBLITS
    global HHBLITS_DB
    global HHM_Outputpath

    HHM_Outputpath = os.path.join(outputpath, 'hhm')
    error_file = os.path.join(os.path.dirname(HHM_Outputpath), error_filename)
    if os.path.exists(error_file):
        os.remove(error_file)
    Fasta_Outputpath = os.path.join(outputpath, 'fasta')
    dssppath = inputpath
    if not os.path.exists(HHM_Outputpath):
        os.makedirs(HHM_Outputpath)
    if not os.path.exists(Fasta_Outputpath):
        os.makedirs(Fasta_Outputpath)
    
    if name_list != '':
        with open(name_list, 'r') as f:
            prots = f.read().strip().split()
    else:
        prots = get_name_list(dssppath)

    print('start')


    args = [(os.path.join(dssppath, prot+'.dssp'),
                os.path.join(Fasta_Outputpath, prot+'.fasta'),
                os.path.join(HHM_Outputpath, prot+'.hhm'))
                for prot in prots]
    with Pool(processers) as pool:
        for arg in args:
            pool.apply_async(pool_generate_hhm, args=(arg,))  # 生成hhm
        pool.close()
        pool.join()
    print('finished', error_file)
    return HHM_Outputpath

def pool_generate_hhm(arg):
    generate_fasta_and_hhm(arg[0], arg[1], arg[2])

def get_fasta(dssp_file, fasta_file):
    prot_name = os.path.splitext(dssp_file.split('/')[-1])[0]
    fastatext = '>' + prot_name + '\n' + getFasta(dssp_file)
    with open(fasta_file, 'w') as f:
        f.writelines(fastatext)

def generate_fasta_and_hhm(dssp_file, fasta_file, hhm_file):
    get_fasta(dssp_file, fasta_file)
    run_generate_hhm(hhm_file, fasta_file)

def run_generate_hhm(hhm_file, fasta_file):
    # 参数设置
    outfmt_type = 5
    num_iter = 10
    evalue_threshold = 0.001
    global HHBLITS
    global HHBLITS_DB

    if os.path.exists(hhm_file):
        return
    cmd = ' '.join([HHBLITS,
                    '-i ' + fasta_file,
                    '-d ' + HHBLITS_DB,
                    # '-out ' + '/dev/null',  # 不显示xml信息
                    # '-e ' + str(evalue_threshold),
                    '-n ' + str(num_iter),
                    # '-outfmt ' + str(outfmt_type),
                    '-ohhm ' + hhm_file,  # Write the hhm file
                    '-hide_cons',
                    '-hide_pred ',
                    '-v 0',
                    '-norealign ',
                    '-o /dev/null',
                    '-cpu ' + '8'
                    ,'-M first'
                    , ' > /dev/null']
                    )
    
    try:
        st = time.time()
        info = subprocess.call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
    global HHM_Outputpath

def arg_parse():
    global error_filename
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inputpath', '-i', type=str, default="", required=True, help="a path of input dssp files")
    parser.add_argument('--list', '-l', type=str, default="", required=False, help="a protname list, join with space")
    parser.add_argument('--outputpath', '-o', type=str, default="", required=False,
                        help="a path for saving related output")
    parser.add_argument('--errorfile_comment', '-ec', type=str, default="", required=False)
    parser.add_argument('--processers', '-p', type=int, default=8, required=False)
    args = parser.parse_args()

    input_path = args.inputpath
    output_path = args.outputpath if args.outputpath != "" else os.path.join(input_path, 'output')
    if args.errorfile_comment != "":
        error_filename = f'no_hhm_fasta_{args.errorfile_comment}.txt'
    else:
        error_filename = f'no_hhm_fasta_{args.errorfile_comment}.txt'
    main(input_path, output_path, name_list=args.list, processers=args.processers)

if __name__=='__main__':
    arg_parse()
    