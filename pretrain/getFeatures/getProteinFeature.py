'''获取单个蛋白质片段特征'''
import os
from tracemalloc import start

import numpy as np
import copy
from multiprocessing import Pool

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
            if residue_id == '' or AA == 'X':  # 跳过注释行
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
                        # cnt_blank = 0
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
                print('Unknown structure:', structure)
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
    else:   # 无结构信息的行数大于等于3或无法合并到上一个，单独成块
        wid += 1
        result[wid] = {}
        result[wid]['list'] = blanklist['list']
        result[wid]['seq'] = blanklist['seq']
        result[wid]['structure'] = 'C'  # 结构为空
        blanklist = copy.deepcopy(initBlank())
        cnt_blank = 0
    return result



def getCompositionFeature(dsspres):
    res = []
    for item in dsspres.values():
        seq = item['seq']
        if len(seq) == 0:
            break
        dict = ['A', 'R', 'N', 'D', 'C',
                'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P',
                'S', 'T', 'W', 'Y', 'V']
        cnt = np.zeros(20)
        for AA in seq:
            for i in range(20):
                if AA == dict[i]:
                    cnt[i] += 1
        feature = cnt / len(seq)
        res.append(feature)
    return np.array(res)

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
        if len(item['seq']) == 0:
            break
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
            if length != 0:
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

    return norss

def get_norm_stru_feature(dsspfile, dsspres):
    stru_info = norm_ss(dsspfile)
    start = 0
    res = []
    stru_info = np.array(list(stru_info.values()))
    for item in dsspres.values():
        seqlist = item['seq']
        length = len(seqlist)
        if length == 0:
            break
        if(start+length <= len(stru_info)):
            tmp = stru_info[start:start+length]
        else:
            print('structure info error：', dsspfile)
            print('info len:', len(stru_info), 'want len:', start+length)
            break
        stru_feature = np.sum(tmp, axis=0) / len(tmp) 
        # print(stru_feature.shape)
        stru_feature = stru_feature.reshape(6)
        # print('after reshape', stru_feature.shape)

        start += length
        res.append(stru_feature)
    res = np.array(res)
    return res

is_import_structure_feature = False
is_import_new_str = True
def getFeatures(hhmfile, import_structure_feature=is_import_structure_feature, import_new_str=is_import_new_str, is_filter=False):
    global dssppath
    global pssmfile
    global feature_path
    global FEATURE_VERSION
    prot_name = os.path.split(hhmfile)[1].split('.')[0]
    dsspfile = os.path.join(dssppath,  prot_name + '.dssp')
    pssmfile = os.path.join(pssmpath,  prot_name + '.pssm')    
    
    try:
        res = splitDsspData(dsspfile)

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
            stru_infor_feature = get_norm_stru_feature(dsspfile, res)
            try:
                structure_feature = np.concatenate((structure_type, stru_infor_feature), axis=1)
            except:

                print('structure concat error shape dismatch', stru_infor_feature.shape, structure_type.shape)
        

        if import_structure_feature:
            import getStructureFeature
            from getStructureFeature import get_structure_feature  
            trained_structure_feature = get_structure_feature(dsspfile, filter_flag=False)
            try:
                structure_feature = np.concatenate((structure_feature, trained_structure_feature), axis=1)
            except:
                print('structure concat error shape dismatch', trained_structure_feature.shape, structure_type.shape)

        try:
            feature = np.concatenate((evolution_feature, composition_feature), axis=1)
        except:
            print('evolution_feature error:', dsspfile)
            print(f'pssm：{pssm_feature.shape} hhm: {hhm_feature.shape} right:{composition_feature.shape}' )
            return

        feature = np.concatenate((feature, structure_feature), axis=1)

        
        # 过滤太短或太长的序列
        tmp = feature.copy()
        if is_filter:
            filtered_keys = filter(res)
            # 剔除太短或太长的序列特征
            for item in reversed(filtered_keys):  # 从后往前删key
                feature = np.delete(feature, item - 1, axis=0)

        # 特征被过滤空处理
        save = True
        if np.all(feature):
            if tmp.size != 0:
                feature = tmp
                print('come back', hhmfile.split('/')[-1])
            else: # 生成特征为空不保存
                save = False
                print(
                    f"no feature: {hhmfile.split('/')[-1]} pssm: {pssm_feature.shape} hhm: {hhm_feature.shape} com: {composition_feature.shape} ")
        else:
            save = False

        feature_filename = os.path.join(feature_path, prot_name + '.txt')

        if save:           
            np.savetxt(feature_filename, feature, delimiter=',')

    except Exception as e:
        
        import traceback
        print('=================================')
        print('getProteinFeature_getFeatures error', dsspfile, repr(e))
        traceback.print_exc()
        # exit()
        print('---------------------------------')

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
        if length == 0:
            break
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
        res.append(hhm_feature)
    res = np.array(res)
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
        lines = lines[:endline]
        
        try:
            matrix = np.array([line.split()[2:22] for line in lines], dtype=float)
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

    for i in range(len(matrix)):
        for j in range(20):
            matrix[i][j] = 1 / ( 1 + np.exp(-matrix[i][j]) )

    start = 0
    res = []

    for item in dsspres.values():
        seqlist = item['seq']
        length = len(seqlist)
        if length == 0:
            break
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


feature_path = ''
dssppath = ''
pssmpath = ''

def main(input_path, dssp_path='', hhm_path='', pssm_path='', name_list=''):
    global dssppath
    global pssmpath
    global feature_path

    # 输入的路径下需包括 dssp 与 pssm 两个目录
    dssppath = os.path.join(input_path, 'dssp') if dssp_path == '' else dssp_path
    hhmpath = os.path.join(input_path, 'hhm') if hhm_path == '' else hhm_path
    pssmpath = os.path.join(input_path, 'pssm') if pssm_path == '' else pssm_path
    feature_path = os.path.join(input_path, 'feature')
    print(feature_path, dssp_path, hhm_path, pssm_path)

    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    pool = Pool(18)

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

    args = parser.parse_args()
    main(args.input, args.dssp, args.hhm, args.pssm)

if __name__ == '__main__':
  
    dataset = 'train'
    script()

    pass