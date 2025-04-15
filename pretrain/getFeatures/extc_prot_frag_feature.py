'''获取单个蛋白质片段特征 80维'''
import os
import pdb
import numpy as np
import copy
from multiprocessing import Pool

import warnings

warnings.filterwarnings("ignore")

cnt_dssp = [0]
def initBlank():
    set = {}
    set['list'] = []
    set['seq'] = []
    return set

def splitDsspData(dsspfile):
    with open(dsspfile, 'r') as f:
        dssp_text = f.readlines()
    alltypes = ['H', 'B', 'S', 'T', 'G', 'I', 'E']
    maxASA = {'G': 188, 'A': 198, 'V': 220, 'I': 233, 'L': 304, 'F': 272, 'P': 203, 'M': 262, 'W': 317, 'C': 201,
              'S': 234, 'T': 215, 'N': 254, 'Q': 259, 'Y': 304, 'H': 258, 'D': 236, 'E': 262, 'K': 317, 'R': 319}
    specailtypes = ['S', 'T', 'B']
    result = {}  # 存放信息：'list' / 'structure' / 'sequence'
    wid = 0
    process_flag = False  # 标志是否枚举到原子信息
    last_structure = '#'  # 表示刚开始
    cnt_blank = 0
    blanklist = copy.deepcopy(initBlank())
    cnt_unk = 0
    sumlines = 0
    cnt_line = -1
    for line in dssp_text:
        cnt_line += 1
        tmp = line.strip()
        if tmp.startswith('#'):
            process_flag = True
            continue
    
        if process_flag:
            
            residue_id = line[7:11].strip()
            # if len(residue_id) == 3:
            #     print(residue_id)
            structure = line[16]
            AA = line[13]
            # print(AA)
            if residue_id == '' :  # 跳过注释行
                # if line[13] not in maxASA.keys(): # 会跳过X
                #     print(line)
                # blanklist = copy.deepcopy(initBlank())
                # cnt_blank = 0
                cnt_unk += 1
                continue
            
            sumlines += 1
            if structure in alltypes:
                if cnt_blank != 0 and cnt_blank < 3:
                    if (wid in result.keys() and result[wid]['structure'] in specailtypes):
                        # print(tmp)
                        result[wid]['list'] += blanklist['list']
                        last_structure = result[wid]['structure']
                        result[wid]['seq'] += blanklist['seq']
                        blanklist = copy.deepcopy(initBlank())
                        cnt_blank = 0
                    elif structure not in specailtypes:
                        wid += 1
                        result[wid] = {}
                        result[wid]['list'] = blanklist['list']
                        result[wid]['structure'] = 'C'  # 结构为空
                        result[wid]['seq'] = blanklist['seq']
                        blanklist = copy.deepcopy(initBlank())
                        cnt_blank = 0
                    # 若不能与新链与老链拼接，则新创建链
                elif cnt_blank > 0:  # 无结构信息的行数大于3或无法与新结构拼接，单独成块
                    wid += 1
                    result[wid] = {}
                    result[wid]['list'] = blanklist['list']
                    result[wid]['structure'] = 'C'  # 结构为空
                    result[wid]['seq'] = blanklist['seq']
                    blanklist = copy.deepcopy(initBlank())
                    cnt_blank = 0

                if structure == last_structure:
                    result[wid]['list'].append(residue_id)
                    result[wid]['seq'].append(AA)
                else:  # 新链
                    wid += 1

                    if cnt_blank != 0 and cnt_blank < 3:  # 插入到新链的空白行
                        result[wid] = {}
                        result[wid]['list'] = blanklist['list']
                        result[wid]['seq'] = blanklist['seq']
                    if wid not in result.keys():
                        result[wid] = {}
                        result[wid]['list'] = [residue_id]
                        result[wid]['seq'] = [AA]
                    else:
                        result[wid]['list'].append(residue_id)
                        result[wid]['seq'].append(AA)
                    result[wid]['structure'] = structure
                blanklist = copy.deepcopy(initBlank())
                cnt_blank = 0
            elif structure == ' ':
                cnt_blank += 1
                blanklist['list'].append(residue_id)
                blanklist['seq'].append(AA)
            else:
                # print('Unknown structure:', structure)
                pass
            last_structure = structure
    # 空白结构结尾
    if cnt_blank != 0 and cnt_blank < 3 and result[wid]['structure'] in specailtypes:  # 划分到上一个结构块
        # print(tmp)
        result[wid]['list'] += blanklist['list']
        result[wid]['seq'] += blanklist['seq']
        last_structure = result[wid]['structure']
        blanklist = copy.deepcopy(initBlank())
        cnt_blank = 0
    elif cnt_blank >= 3:   # 无结构信息的行数大于等于3或无法合并到上一个，单独成块
        wid += 1
        result[wid] = {}
        result[wid]['list'] = blanklist['list']
        result[wid]['seq'] = blanklist['seq']
        result[wid]['structure'] = 'C'  # 结构为空
        blanklist = copy.deepcopy(initBlank())
        cnt_blank = 0
    return result


def saveSequences(results):
    seq = ''
    text = ''
    cnt = 0
    for item in results.values():
        seq = ''.join(item['seq'])
        text += '>' + str(cnt) + '\n' + seq + '\n'
        cnt += 1
    filename = 'test.txt'
    with open(filename, 'w') as f:
        f.writelines(text)

def getCompositionFeature(dsspres):
    res = np.zeros((0, 20))
    for item in dsspres.values():
        seq = item['seq']
        dict = ['A', 'R', 'N', 'D', 'C',
                'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P',
                'S', 'T', 'W', 'Y', 'V']
        cnt = np.zeros(20)
        for AA in seq:
            for i in range(20):
                if AA == dict[i]:
                    cnt[i] += 1
        tmp = cnt / len(seq)
        res = np.vstack([res, tmp])
    return res #

def getOneHot(dsspres):
    res = np.zeros((0, 8))
    dict = {'H':['0', '0', '0', '0', '0', '0', '0', '1'],
            'B':['0', '0', '0', '0', '0', '0', '1', '0'],
            'E':['0', '0', '0', '0', '0', '1', '0', '0'],
            'G':['0', '0', '0', '0', '1', '0', '0', '0'],
            'I':['0', '0', '0', '1', '0', '0', '0', '0'],
            'T':['0', '0', '1', '0', '0', '0', '0', '0'],
            'S':['0', '1', '0', '0', '0', '0', '0', '0'],
            'C':['1', '0', '0', '0', '0', '0', '0', '0'],
            }
    for item in dsspres.values():
        AA = item['structure']
        res = np.vstack([res, np.array(dict[AA], dtype=int)])
    # print('shape4:', res.shape)
    return res

def filter(res):
    res_cp = res.copy()
    ret = []
    for key, value in res_cp.items():
        seq = value['seq']
        length = len(seq)
        if length < 3 or length > 60:
            ret.append(key)
    # print(ret)
    return ret

def norm_ss(dssp_path):
    maxASA = {'G': 188, 'A': 198, 'V': 220, 'I': 233, 'L': 304, 'F': 272, 'P': 203, 'M': 262, 'W': 317, 'C': 201,
              'S': 234, 'T': 215, 'N': 254, 'Q': 259, 'Y': 304, 'H': 258, 'D': 236, 'E': 262, 'K': 317, 'R': 319, 'X': 0.1}
    map_ss_8 = {' ': [1, 0, 0, 0, 0, 0, 0, 0], 'S': [0, 1, 0, 0, 0, 0, 0, 0], 'T': [0, 0, 1, 0, 0, 0, 0, 0],
                'H': [0, 0, 0, 1, 0, 0, 0, 0],
                'G': [0, 0, 0, 0, 1, 0, 0, 0], 'I': [0, 0, 0, 0, 0, 1, 0, 0], 'E': [0, 0, 0, 0, 0, 0, 1, 0],
                'B': [0, 0, 0, 0, 0, 0, 0, 1]}

    with open(dssp_path,'r') as f:
        text = f.readlines()
    
    prot_name = os.path.split(dssp_path)[1].split('.')[0]
    
    start_line = 0
    for i in range(0, len(text)):
        if text[i].strip().startswith('#'):
            start_line = i + 1
            
            break
    
    norss = {}
    for i in range(start_line, len(text)):
        line = text[i]
        # print(line)
        residue_id = line[5:11].strip()
        if line[13] not in maxASA.keys() or residue_id == '':
            continue

        res_dssp = np.zeros([6])
        res_dssp[0] = min(float(line[35:38]) / maxASA[line[13]], 1)
        res_dssp[1] = (float(line[85:91]) + 1) / 2
        res_dssp[2] = min(1, float(line[91:97]) / 180)
        res_dssp[3] = min(1, (float(line[97:103]) + 180) / 360)
        res_dssp[4] = min(1, (float(line[103:109]) + 180) / 360)
        res_dssp[5] = min(1, (float(line[109:115]) + 180) / 360)
        norss[residue_id] = res_dssp.reshape((1, -1))
    # print(norss)
   
    return norss

def get_norm_stru_feature(dsspfile, dsspres):
    stru_info = norm_ss(dsspfile)
    start = 0
    res = []
    stru_info = np.array(list(stru_info.values()))
    for item in dsspres.values():
        seqlist = item['seq']
        length = len(seqlist)

        if(start+length <= len(stru_info)):
            tmp = stru_info[start:start+length]
        else:
            print('structure info error：', dsspfile)
            print('info len:', len(stru_info), 'want len:', start+length)
            break
        stru_feature = np.sum(tmp, axis=0) / len(tmp) # pssm_feature.shape : (20, )
        stru_feature = stru_feature.reshape(6)
        start += length
        res.append(stru_feature)
    res = np.array(res)
    return res

import numpy as np

# 只加入了20个列
def get_20_hhm(hhmfile, dsspres):
    with open(hhmfile, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    num_lines = len(lines)
    
    i = 0
    flag = False
    hhm = []
    tmp_s = 0
    while i < num_lines:
        if not flag:
            if not lines[i].startswith('#'):
                i += 1
            else:
                flag = True
                tmp_s = i
                i += 5
                continue
        else:
            if lines[i].startswith('//'):
                
                break
            tmp = []
            strs = lines[i].split()[2:-1]
            # print(strs)
            for str in strs:
                if str == '*':
                    tmp.append(1)
                else:
                    tmp.append(int(str) / 10000)
            # print(len(tmp), tmp)
            hhm.append(tmp)
            i += 3

    hhm = np.array(hhm, dtype=float)
    start = 0
    res = []
    sum = 0
    for item in dsspres.values():
        seqlist = item['seq']
        length = len(seqlist)
        sum += len(seqlist)
        if (start + length <= len(hhm)):
            tmp = hhm[start:start + length]
        elif (start < len(hhm)):
            tmp = hhm[start:]
        else:
            print('len match error：', hhmfile)
            print(f'hhm len: {len(hhm)}, dssp len:{start + length}')
            break
        hhm_feature = np.sum(tmp, axis=0) / len(tmp)  # pssm_feature.shape : (20, )
        if (np.isnan(hhm_feature[0])):
            print('nan error 1：', hhmfile)
            print('hhm fragment 1', tmp)
            print('seq list', seqlist)

        hhm_feature.reshape(1, 20)
        start += length
        # print(len(tmp), tmp.shape)
        # pprint.pprint(tmp)
        # print(np.sum(tmp, axis=0))
        res.append(hhm_feature)
    # print(sum)
    res = np.array(res)
        # print('pssm len', len(pssm))
        # print('fragement pssm', tmp)
        # print('sum:', np.sum(tmp, axis=0) )
        # print('tmp', np.sum(tmp, axis=0) / len(tmp) )
        # print('------------')

        # print('shape1:', res.shape)
    return res

def get_pssm_feature(filepath, dsspres):
    # 读取矩阵    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()[3:]
        endline = len(lines)
        
        for i in range(endline - 10, endline):
            if lines[i].strip() == "":
                endline = i
                break
        # print('seq len:', endline)
        lines = lines[:endline]
        
        try:
            matrix = np.array([line.split()[2:22] for line in lines], dtype=float)
            # seq = [line.split()[0:2] for line in lines]
        except Exception as e:
            print(str(e))
            print('last line error:', filepath)
            with open('pssm_error.txt', 'a') as f:
                prot_name = filepath.split('/')[-1].split('.')[0]
                f.write(prot_name + '\n')

            return
    else:
        filepath = filepath.split('.')[0] + '.npy'
        try:
            matrix = np.loadtxt(filepath)
        except Exception as e:
            print('no pssm or blosum:', filepath)
            return

    # 归一化
    for i in range(len(matrix)):
        for j in range(20):
            matrix[i][j] = 1 / ( 1 + np.exp(-matrix[i][j]) )
    # print(pssm)
    start = 0
    res = []

    for item in dsspres.values():
        seqlist = item['seq']
        length = len(seqlist)

        if(start+length <= len(matrix)):
            tmp = matrix[start:start+length]
        else:
            print('nan error：', filepath)
            print('pssm fragment len:', start+length, 'want len:', length)

            break
        pssm_feature = np.sum(tmp, axis=0) / len(tmp) # pssm_feature.shape : (20, )
        if(np.isnan(pssm_feature[0])):
            print('nan error：', filepath)
            print('pssm fragment', tmp)
            print('seq list', seqlist)

        pssm_feature.reshape(1, 20)
        start += length
        res.append(pssm_feature)
    res = np.array(res)
    return res

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import HSExposureCB, HSExposureCA, Selection, ResidueDepth
from Bio.PDB.ResidueDepth import residue_depth, get_surface, ca_depth
def get_solvent_feature(pdb_file, dsspres):
    if feature_path == '/home2/xeweng/data/pdb_chains/solvent/feature/feature':
        prot_name = os.path.split(pdb_file)[1].split('.')[0]
        if len(prot_name.split('_')[1]) == 2:
            prot_name = prot_name.split('_')[0] + '_' + prot_name.split('_')[1][0].lower()
        pdb_file = os.path.join(pdbpath, prot_name + '.pdb')
    pdb_hse_feature = []
    
    # 先提取
    parser = PDBParser()
    structure = parser.get_structure("x", pdb_file)
    model = structure[0]
    try:
        hse_a = HSExposureCA(model)
    except:
        print('HSExposureCA error:', pdb_file)

    residue_list = Selection.unfold_entities(model,'R')

    try:
        hse_b = HSExposureCB(model)
    except:
        print('HSExposureCB error:', pdb_file)

    residue_list = Selection.unfold_entities(model,'R')

    AA = ['GLY','ALA','VAL','LEU','ILE','PHE','TRP','TYR','ASP','ASN',
      'GLU','LYS','GLN','MET','SER','THR','CYS','PRO','HIS','ARG'] # 会滤过未知残基无法与未过滤的对齐

    try:
        surface = get_surface(model)
    except:
        print('get_surface error:', pdb_file)
        surface = None
    for r in residue_list[:]:
        hse_feature = [0] * 6
        # ARG {'EXP_HSE_A_U': 10, 'EXP_HSE_A_D': 20, 'EXP_CB_PCB_ANGLE': 0.4065649357658556}
        if r.get_resname() not in AA:
            continue
        if len(r.xtra.keys()) != 0:
            if 'EXP_HSE_A_U' in r.xtra.keys():
                hse_feature[0] = r.xtra['EXP_HSE_A_U'] / 60
                hse_feature[1] = r.xtra['EXP_HSE_A_D'] / 60
            if 'EXP_HSE_B_U' in r.xtra.keys():
                hse_feature[2] = r.xtra['EXP_HSE_B_U'] / 60
                hse_feature[3] = r.xtra['EXP_HSE_B_D'] / 60
            if surface is None:
                hse_feature[4] = 1.00
                hse_feature[5] = 1.00
            else:
                hse_feature[4] = ca_depth(r, surface) / 10
                hse_feature[5] = residue_depth(r, surface) / 10
        pdb_hse_feature.append(hse_feature)
    
    # print(pdb_hse_feature[:5])
    pdb_hse_feature = np.array(pdb_hse_feature)

    # 再根据片段切分
    start = 0
    res = []
    for item in dsspres.values():
        seqlist = item['seq']
        length = len(seqlist)

        if(start+length <= len(pdb_hse_feature)):
            tmp = pdb_hse_feature[start:start+length]
        else:
            tmp = pdb_hse_feature[start:]
            # print('solvent len error：', pdb_file)
            # print('solvent fragment len:', len(pdb_hse_feature), 'want len:', start + length)

            # break
        hse_feature = np.sum(tmp, axis=0) / len(tmp) # pssm_feature.shape : (20, )
        if(np.isnan(hse_feature).any()):
            hse_feature[0] = 0
            hse_feature[1] = 0
            hse_feature[2] = 0
            hse_feature[3] = 0
            hse_feature[4] = 1.99
            hse_feature[5] = 1.99
        hse_feature.reshape(1, 6)
        start += length
        res.append(hse_feature)
    res = np.array(res)
    return res


def getFeatures(hhmfile, import_structure_feature=False, import_new_str=True):
    global dssppath
    global pssmfile
    global feature_path
    global pdbpath
    global FEATURE_VERSION
    prot_name = os.path.split(hhmfile)[1].split('.')[0]
    dsspfile = os.path.join(dssppath,  prot_name + '.dssp')
    pssmfile = os.path.join(pssmpath,  prot_name + '.pssm')
    pdbfile = os.path.join(pdbpath, prot_name + '.pdb')
    
    feature_filename = os.path.join(feature_path, prot_name + '.txt')
    # # 跳过已经生成的
    # if os.path.exists(feature_filename): 
    #     return
    
    
    # # 查看结构切分是否正确
    # sum = 0
    # for item in res.values():
    #     sum += len(item['seq'])
    #     print(item['list'], item['structure'])
    # print(sum)

    try:
        res = splitDsspData(dsspfile)
        # print(res)
        
        

        # 特征提取    
        pssm_feature = get_pssm_feature(pssmfile, res)
        hhm_feature = get_20_hhm(hhmfile, res)
        composition_feature = getCompositionFeature(res)
        
        
        structure_type = getOneHot(res)
        try:
            evolution_feature = np.concatenate((pssm_feature, hhm_feature), axis=1)
        except Exception as e:
            print('evolution error:', dsspfile)
            with open('hhm_error_dssp.txt', 'a') as f:
                f.write(dsspfile + '\n')
            print(f'pssm：{pssm_feature.shape} hhm: {hhm_feature.shape} right:{composition_feature.shape}' )
            print(e)
            return
        
        # 特征合并
        structure_feature = structure_type
        if import_new_str:
            solvent_feature = get_solvent_feature(pdbfile, res)
            stru_infor_feature = get_norm_stru_feature(dsspfile, res)
            try:
                structure_feature = np.concatenate((structure_type, stru_infor_feature), axis=1)
                if solvent_feature is None:
                    print(solvent_feature)
                    print('solvent error:', dsspfile)
                    return
                structure_feature = np.concatenate((structure_feature, solvent_feature), axis=1)
            except:
                # print(res)
                # sum = 0
                # appear = {}
                # for item in res.values():
                #     sum += len(item['seq'])
                #     for res_id in item['list']:
                #         if int(res_id) in appear.keys():
                #             print('appear error:', dsspfile, res_id, item['list'], appear[int(res_id)])
                            
                #         appear[int(res_id)] = item['list']
                #         if int(res_id) not in struc_in.keys():
                #             print('struc_in error:', dsspfile, res_id)
                # struc_in = norm_ss(dsspfile)
                # print(sum, len(struc_in.keys()))    
                print('structure concat error shape dismatch', stru_infor_feature.shape, structure_type.shape, solvent_feature.shape)
                return
        

        if import_structure_feature:
            import getStructureFeature
            from getStructureFeature import get_structure_feature  
            trained_structure_feature = get_structure_feature(dsspfile, filter_flag=False)
            try:
                structure_feature = np.concatenate((structure_feature, trained_structure_feature), axis=1)
            except:
                print('structure concat error shape dismatch', trained_structure_feature.shape, structure_type.shape)

        feature = np.concatenate((evolution_feature, composition_feature), axis=1)
        feature = np.concatenate((feature, structure_feature), axis=1)
        # print(feature.shape)
      
        
        # 过滤太短或太长的序列
        filtered_keys = filter(res)
        tmp = feature.copy()
        # 剔除太短或太长的序列特征
        for item in reversed(filtered_keys):  # 从后往前删key
            feature = np.delete(feature, item - 1, axis=0)

        # 特征被过滤空处理
        
        save = True
        # 生成特征为空不保存
        if np.all(feature):
            # print(res)
            # print(pssmfile)
            if tmp.size != 0:
                feature = tmp
                print('come back', hhmfile.split('/')[-1])
            else:
                save = False
                print(
                    f"no feature: {hhmfile.split('/')[-1]} pssm: {pssm_feature.shape} hhm: {hhm_feature.shape} com: {composition_feature.shape} ")

        # 特征保存
        # print('shape3:', feature.shape)
        feature_filename = os.path.join(feature_path, prot_name + '.txt')

        if save:           
            np.savetxt(feature_filename, feature, delimiter=',')

            # 以片段存表征
            # for index, frag in enumerate(feature):
            #     frag_filename = os.path.join(feature_path, prot_name + f'-frag{index}.txt')
            #     np.savetxt(frag_filename, frag, delimiter=',')
    except Exception as e:
        
        import traceback
        traceback.print_exc()
        print('getProteinFeature_getFeatures error', dsspfile, repr(e))
        # exit()

# global 变量
feature_path = ''
dssppath = ''
pssmpath = ''
pdbpath = ''
def main(input_path, dssp_path='', hhm_path='', pssm_path='', pdb_path=''):
    global dssppath
    global pssmpath
    global feature_path
    global pdbpath

    # 输入的路径下需包括 dssp 与 pssm 两个目录
    dssppath = os.path.join(input_path, 'dssp') if dssp_path == '' else dssp_path
    hhmpath = os.path.join(input_path, 'hhm') if hhm_path == '' else hhm_path
    pssmpath = os.path.join(input_path, 'pssm') if pssm_path == '' else pssm_path
    pdbpath = pdb_path
    feature_path = os.path.join(input_path, 'feature')
    print(feature_path, dssp_path, hhm_path, pssm_path)

    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    pool = Pool(128)

    # 使用列表推导式获取文件夹下所有文件的绝对路径列表
    file_paths = [os.path.join(root, file_name) for root, _, files in os.walk(hhm_path) for file_name in files]
    pool.map(getFeatures, file_paths)
    pool.close()
    pool.join()

    return feature_path

def script():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='输入文件夹路径')
    parser.add_argument('-d', '--dssp', help='dssp文件夹路径')
    parser.add_argument('-p', '--pssm', help='pssm文件夹路径')
    parser.add_argument('-hh', '--hhm', help='hhm文件夹路径')
    parser.add_argument('-b', '--pdb', required=False, help='pdb文件夹路径')
    args = parser.parse_args()
    main(args.input, args.dssp, args.hhm, args.pssm, args.pdb)

if __name__ == '__main__':
    # debug 专区
    # main()
    # hhm = "/home2/xeweng/forRemote/downstream/_GO/data/hhm_output/test/hhm/1K0K-A.hhm"
    # get_30_hhm(hhm, '')
    dataset = 'train'
    # dssppath = f"/home2/xeweng/forRemote/downstream/_GO/data/hhm_output/{dataset}/dssp"
    # hhmpath = f"/home2/xeweng/forRemote/downstream/_GO/data/hhm_output/{dataset}/hhm"
    # pssmpath = f'/home2/xeweng/forRemote/downstream/_GO/big_data/data/pssm/{dataset}/pssm' 
    # pdbpath = f'/home2/xeweng/forRemote/downstream/_GO/data/{dataset}/'
    # featurepath = f"/home2/xeweng/data/pdb_chains/solvent/go/{dataset}/" 
    # featurepath = f'/home2/xeweng/forRemote/downstream/_GO/data/hhm_output/{dataset}/feature'


    dssppath = f"/home2/xeweng/data/pdb_chains/hhm_output/dssp"
    pssmpath = f'/home2/xeweng/data/pdb_chains/pssm_output/pssm'  
    hhmpath = f'/home2/xeweng/data/pdb_chains/hhm_output/hhm'
    # hhmfile = '/home2/xeweng/data/pdb_chains/hhm_output/hhm/6GMH_V.hhm'
    pdbpath = '/home2/xeweng/data/pdb_chains/PROT/pdb'
    featurepath = '/home2/xeweng/data/pdb_chains/solvent/feature'
    
    files = os.listdir(hhmpath)
    # getFeatures(hhmfile)
    # print(cnt_dssp[0], cnt_dssp[0] / len(files))
    main(featurepath, dssppath, hhmpath, pssmpath, pdbpath)
    print(len(os.listdir(os.path.join(featurepath, 'feature'))))
    # for file in files[:]:
    #     hhmfile = os.path.join(hhmpath, file)
    #     getFeatures(hhmfile)

    # 脚本化
    # script()

    pass