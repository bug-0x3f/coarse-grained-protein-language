from multiprocessing import Pool
import os, time
import subprocess
import tempfile, shutil

from torch import le


OUTPUT_PATH = ''
error_file = None
def multi_generate_dssp(arg):
    pdb_file_path, dssp_file_path = arg
    if os.path.exists(dssp_file_path): # 跳过已经生成的
        if(len(open(dssp_file_path, 'r').readlines()) > 2):
            return
    command = ["dssp", "-i", pdb_file_path, "-o", dssp_file_path]
    global OUTPUT_PATH
    try:
        res = subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # with open(os.path.join(OUTPUT_PATH, f'dssp_error_{timestamp}.log'), 'a') as f:
        #     f.write(dssp_file_path + '\n' + str(e.stderr.decode('utf-8')) + '\n\n')
        with open(error_file, 'a') as f:
            f.write(pdb_file_path + '\n')
        print(str(e.stderr.decode('utf-8')))
        
        global error_dssp
     
        print(f"生成 {dssp_file_path} 时出错: {e}")

def simplify_chain_id(pdb_file_path):
    """
    简化PDB文件中的双字符链ID为单字符
    
    参数:
    pdb_file_path: str, 原始PDB文件的路径
    output_file_path: str, 修改后的PDB文件的输出路径
    """
    
    # 创建一个临时文件
    temp_fd, temp_path = tempfile.mkstemp()
    with os.fdopen(temp_fd, 'w') as outfile, open(pdb_file_path, 'r') as infile:     
        for line in infile:
            # PDB文件中定义链ID的字段在22位（从0开始计数），对于ATOM和HETATM记录行
            if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                # 将双字符链ID简化为单字符
                simplified_chain_id = line[21].upper()  # 选取首字符，并确保为大写
                newline = line[:21] + simplified_chain_id + line[23:]
                outfile.write(newline)
            else:
                # 对于不需要修改的行，直接写入输出文件
                outfile.write(line)
    
    shutil.move(temp_path, pdb_file_path)
    # print(temp_path) # 临时文件存在 /tmp

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
        simplify_chain_id(pdb_file)
        structure = parser.get_structure(pid, pdb_file)
        
    
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
                    # print str(residue.get_resname()).strip()
                    if (str(residue.get_resname()).strip() in DA) or (str(residue.get_resname()).strip() in RA):
                        for atom in residue:
                            atom_list.append(atom)
                    else:
                        pass
            n += 1

    seq = ''.join(residues)
    return seq



def main(input_path, output_path,  name_list=[]):
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError("生成dssp错误：输入指定的路径不存在 " + input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    print("-----generate dssp start-----")
    # 遍历指定目录下的文件
    global OUTPUT_PATH
    OUTPUT_PATH = output_path
    global error_file
    error_file = os.path.join(os.path.dirname(OUTPUT_PATH), f'no_dssp_pdb_{comment}.txt')
    print('dssp error path',error_file)
    if os.path.exists(error_file):
        os.remove(error_file)
    pdb_list = []
    dssp_list = []
    if name_list == []:
        for file in os.listdir(input_path):
            prot_name = file.split('.')[0]
            if file.endswith(".pdb"):
                pdb_file_path = os.path.join(input_path, file)
                dssp_file_path = os.path.join(output_path, os.path.splitext(file)[0] + ".dssp")
                pdb_list.append(pdb_file_path)
                dssp_list.append(dssp_file_path)
    else:
        for prot_name in name_list:
            pdb_file_path = os.path.join(input_path, prot_name +'.pdb')
            dssp_file_path = os.path.join(output_path, prot_name + ".dssp")
            pdb_list.append(pdb_file_path)
            dssp_list.append(dssp_file_path)

    pool = Pool(64)
    params = [arg for arg in zip(pdb_list, dssp_list)]
    pool.map(multi_generate_dssp, params)
    pool.close()
    pool.join()

    return
    # 统一处理缺失dssp的pdb文件
    if os.path.exists(error_file):
        print('准备处理缺失dssp的pdb:', error_file)
        import esm
        import torch
        # 载入esm模型
        model = esm.pretrained.esmfold_v1()
        model.set_chunk_size(64)
        model = model.eval().cuda()
        print("模型载入完成")
        with open(error_file, 'r') as f:
            pdbs = [ path.strip() for path in f.readlines() ]
        
        print(f'esm开始生成pdb，共有{len(pdbs)}条序列待预测结构')
        esm_pdb_list = []
        no_seq_pdb_list = []
        for pdb_file in pdbs:
            prot_name = pdb_file.split('/')[-1].split('.')[0]
            # 1. 通过pdb提取序列
            try:
                seq = get_seq_from_pdb(prot_name, pdb_file) 
            except Exception as e:
                print('获取序列失败：', pdb_file)
                print(str(e))
                no_seq_pdb_list.append(pdb_file + '\n')
                continue # 继续处理下一个
            # 2. 将序列输入esmfold模型，得到pdb           
            output_file = os.path.join(input_path, prot_name + '.pdb')
            esm_pdb_list.append(output_file + '\n')
            if not os.path.exists(output_file):            
                with torch.no_grad():
                    pdb_content = model.infer_pdb(seq)                
                with open(output_file, "w") as fw:
                    fw.write(pdb_content)
            
            # 3. 用新的pdb生成dssp 
            dssp_file_path = os.path.join(output_path, prot_name + ".dssp")
            print(dssp_file_path)
            multi_generate_dssp((output_file, dssp_file_path))
        
        log_file_1 = os.path.join(os.path.dirname(input_path), f'esm_pdb_list_{comment}.txt')
        log_file_2 = os.path.join(os.path.dirname(input_path), f'no_seq_pdb_list_{comment}.txt')
        with open(log_file_1, 'w') as f:
            f.writelines(esm_pdb_list)
        with open(log_file_2, 'w') as f:
            f.writelines(no_seq_pdb_list)
        print('由esm生成的蛋白质名单', log_file_1)
        print('序列获取失败的蛋白质名单', log_file_2)

    print("-----generate dssp done-----")
    print(error_file)

comment = '21w'
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python generate_dssp.py /path/to/pdb/files")
    else:
        input_path = sys.argv[1]
        list_path = f'/home2/xeweng/data/pdb_chains/prot_chain_list_{comment}.txt'
        output_path = '/home2/xeweng/data/pdb_chains/sources/dssp_20w'
        with open(list_path, 'r') as f:
            name_list = f.read().split()
        main(input_path, output_path, name_list)
