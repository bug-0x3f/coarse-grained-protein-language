import os
import numpy as np
from multiprocessing import Pool
from split_structure import split_dssp_data


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
    return ret

def norm_ss(dssp_path):
    maxASA = {'G': 188, 'A': 198, 'V': 220, 'I': 233, 'L': 304, 'F': 272, 'P': 203, 'M': 262, 'W': 317, 'C': 201,
              'S': 234, 'T': 215, 'N': 254, 'Q': 259, 'Y': 304, 'H': 258, 'D': 236, 'E': 262, 'K': 317, 'R': 319, 'X': 0.1}

    with open(dssp_path,'r') as f:
        text = f.readlines()
    
    start_line = 0
    for i in range(0, len(text)):
        if text[i].strip().startswith('#'):
            start_line = i + 1
            break
    
    norss = {}
    for i in range(start_line, len(text)):
        line = text[i]
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
        stru_feature = stru_feature.reshape(6)
        start += length
        res.append(stru_feature)
    res = np.array(res)
    return res

def getFeatures(prot_name):
    global dssppath
    global pssmpath
    global hhmpath
    global feature_path
    dsspfile = os.path.join(dssppath, prot_name + '.dssp')
    pssmfile = os.path.join(pssmpath, prot_name + '.pssm')
    hhmfile = os.path.join(hhmpath, prot_name + '.hhm')    
    
    try:
        res = split_dssp_data(dsspfile)

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
        
        structure_feature = structure_type
        stru_infor_feature = get_norm_stru_feature(dsspfile, res)
        try:
            structure_feature = np.concatenate((structure_type, stru_infor_feature), axis=1)
        except Exception as e:
            print('structure concat error shape dismatch', stru_infor_feature.shape, structure_type.shape)
            print(e)

        try:
            feature = np.concatenate((evolution_feature, composition_feature), axis=1)
        except Exception as e:
            print('evolution_feature error:', dsspfile)
            print(f'pssm：{pssm_feature.shape} hhm: {hhm_feature.shape} right:{composition_feature.shape}' )
            print(e)
            return

        feature = np.concatenate((feature, structure_feature), axis=1)

        tmp = feature.copy()
    
        filtered_keys = filter(res)
        for item in reversed(filtered_keys):
            feature = np.delete(feature, item - 1, axis=0)

        save = True
        if np.all(feature):
            if tmp.size != 0:
                feature = tmp
                print('come back', hhmfile.split('/')[-1])
            else: 
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
        print('---------------------------------')


def get_20_hhm(hhmfile, dsspres):
    with open(hhmfile, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    num_lines = len(lines)
    
    i = 0
    flag = False
    hhm = []
    while i < num_lines:
        if not flag:
            if not lines[i].startswith('#'):
                i += 1
            else:
                flag = True
                i += 5
                continue
        else:
            if lines[i].startswith('//'):
                break
            tmp = []
            strs = lines[i].split()[2:-1]
            for str in strs:
                if str == '*':
                    tmp.append(1)
                else:
                    tmp.append(int(str) / 10000)
            hhm.append(tmp)
            i += 3

    hhm = np.array(hhm, dtype=float)
    start = 0
    res = []
    for item in dsspres.values():
        seqlist = item['seq']
        length = len(seqlist)
        if length == 0:
            break
        if (start + length <= len(hhm)):
            tmp = hhm[start:start + length]
        elif (start < len(hhm)):
            tmp = hhm[start:]
        else:
            print('len match error：', hhmfile)
            print(f'hhm len: {len(hhm)}, dssp len:{start + length}')
            break
        hhm_feature = np.sum(tmp, axis=0) / len(tmp)
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
        pssm_feature = np.sum(tmp, axis=0) / len(tmp)
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
hhmpath = ''

def main(input_path, dssp_path='', hhm_path='', pssm_path='', prots=[]):
    global dssppath
    global pssmpath
    global hhmpath
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
    if not prots:
        file_paths = [os.path.join(root, file_name) for root, _, files in os.walk(hhmpath) for file_name in files]
        prot_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    else:
        prot_names = prots

    pool.map(getFeatures, prot_names)
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
    script()