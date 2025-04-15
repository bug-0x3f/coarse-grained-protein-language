
import os
from multiprocessing import Pool

FEATURE_PATH = ''
DSSP_PATH = ''
PSSM_PATH = ''
HHM_PATH = ''

def generate_features_in_dir(dir_path, dssp_path='', hhm_path='', pssm_path=''):
    global FEATURE_PATH
    global DSSP_PATH
    global PSSM_PATH

    DSSP_PATH = os.path.join(dir_path, 'dssp') if dssp_path == '' else dssp_path
    HHM_PATH = os.path.join(dir_path, 'hhm') if hhm_path == '' else hhm_path
    PSSM_PATH = os.path.join(dir_path, 'pssm') if pssm_path == '' else pssm_path
    FEATURE_PATH = os.path.join(dir_path, 'feature')
    print(FEATURE_PATH, dssp_path, hhm_path, pssm_path)

    if not os.path.exists(FEATURE_PATH):
        os.makedirs(FEATURE_PATH)

    pool = Pool(18)

    prots = [file.split('.')[0] for file in os.listdir(pssm_path)]
    pool.map(get_prot_feature, prots)
    pool.close()
    pool.join()

    return FEATURE_PATH

def get_prot_feature(prot_name):
    global DSSP_PATH
    global PSSM_PATH
    global FEATURE_PATH
    global HHM_PATH
    dsspfile = os.path.join(DSSP_PATH,  prot_name + '.dssp')
    pssmfile = os.path.join(PSSM_PATH,  prot_name + '.pssm')
    hhmfile = os.path.join(HHM_PATH,  prot_name + '.hhm')
    
    
def script():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='输入文件夹路径')
    parser.add_argument('-d', '--dssp', help='dssp文件夹路径')
    parser.add_argument('-p', '--pssm', help='pssm文件夹路径')
    parser.add_argument('-hh', '--hhm', help='hhm文件夹路径')
    parser.add_argument('-o', '--output_dir', help='the path to save ')
    parser.add_argument('-t', '--type', help='single file or all files in directory', default='dir')
    global is_import_new_str
    args = parser.parse_args()
    is_import_new_str = args.new
    if args.type == 'dir':
        generate_features_in_dir(args.input, args.dssp, args.hhm, args.pssm)
    elif args.type == 'file': 
        pass
    else:
        raise ValueError('type must be dir or file')