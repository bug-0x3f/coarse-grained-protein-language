import copy
import numpy as np

def initBlank():
    set = {}
    set['list'] = []
    set['seq'] = []
    return set

def split_dssp_data(dsspfile):
    with open(dsspfile, 'r') as f:
        dssp_text = f.readlines()
    alltypes = ['H', 'B', 'S', 'T', 'G', 'I', 'E']
    specailtypes = ['S', 'T', 'B']
    result = {}  # 存放信息：'list' / 'structure' / 'sequence'
    wid = 0
    process_flag = False  # 标志是否枚举到原子信息
    last_structure = '#'  # 表示刚开始
    cnt_blank = 0
    blanklist = copy.deepcopy(initBlank())
    cnt_unk = 0
    sumlines = 0
    for line in dssp_text:
        tmp = line.strip()
        if tmp.startswith('#'):
            process_flag = True
            continue
        if process_flag:
            sumlines += 1
            residue_id = line[7:10].strip()
            structure = line[16]
            AA = line[13]
            # print(AA)
            if residue_id == '' or residue_id == '!':  # 跳过注释行
                # blanklist = copy.deepcopy(initBlank())
                # cnt_blank = 0
                cnt_unk += 1
                continue

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

def filter_frag(res):
    res_cp = res.copy()
    ret = []
    for key, value in res_cp.items():
        seq = value['seq']
        length = len(seq)
        if length < 3 or length > 60:
            ret.append(key)
    # print(ret)
    return ret

def main(dsspfile, filter=False):
    
    res = split_dssp_data(dsspfile)
    if filter:
        tmp = res.copy()  # 过滤空恢复过滤前的数据
        filtered_keys = filter_frag(res)

        # 剔除太短或太长的片段
        for item in filtered_keys: 
            del res[item]
        if len(res) < 1:
            res = tmp
        
    return res
    
def process_file(file):
    cnt = [0] * 1000
    try:
        res = main(os.path.join(path, file))
        for item in res.values():
            frag_len = len(item['seq'])
            cnt[frag_len] += 1
    except Exception as e:
        print(file)
        print(e)
        # print(res)    
    
    return cnt

def cnt(file):

    try:
        res = main(os.path.join(path, file))
        sentence_len = len(res.values())
        cnt = 0
        if sentence_len > 80:
            for item in res.values():
                frag_len = len(item['seq'])
                if(frag_len < 3):
                    cnt += 1
            print(sentence_len, frag_len)
    except Exception as e:
        print(file)
        print(e)
        # print(res)    
    
    # return cnt

if __name__ == '__main__':
    import os
    path = '/home2/xeweng/data/pdb_chains/hhm_output/dssp'
    files = os.listdir(path)
    
    from multiprocessing import Pool
    # Create a pool of workers
    with Pool(128) as p:
        cnt_list = p.map(cnt, files)

    # Combine the counts from each worker
    # cnt = [sum(x) for x in zip(*cnt_list)]

    # sum = 0
    # for i, num in enumerate(cnt[:318]):
    #     sum += num
    # print(sum)

    # with open('cnt_file_no_filter.txt', 'w') as f:
    #     for item in cnt:
    #         f.write("%s\n" % item)