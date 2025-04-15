#!/usr/bin/env python

import argparse, os
from multiprocessing import Pool
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inputpath', '-i', type=str, default="", required=False, help="a path of input pdb files")
    parser.add_argument('--outputpath', '-o', type=str, default="", required=False,
                        help="a path for saving related output")
    parser.add_argument('-dd', '--dssp_dir', default="", required=False, help='dssp文件夹路径')
    parser.add_argument('-pd', '--pssm_dir', default="", required=False, help='pssm文件夹路径')
    parser.add_argument('-hhd', '--hhm_dir', default="", required=False, help='hhm文件夹路径')
    parser.add_argument('--list', '-l', type=str, default="", required=False, help="a list for select specific pdbs in input_path")

    args = parser.parse_args()


    input_path = args.inputpath
    output_path = os.path.dirname(input_path)
    if input_path.endswith('/'):  # 以斜杠结尾
        output_path = os.path.dirname(output_path)
    output_path = os.path.join(output_path, 'output')
    output_path = args.outputpath if args.outputpath != "" else output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    prots= []
    if args.list != '':
        with open(args.list, 'r') as f:
            prots = f.read().strip().split()
    dataset = args.outputpath.split('/')[-1]
    dssp_path = os.path.join(output_path, 'dssp')
    # dssp_path = "/home2/xeweng/data/pdb_chains/pssm_output/dssp"
    hhm_path = os.path.join(output_path, 'hhm')
    pssm_path = os.path.join(output_path, 'pssm')

    dssp_path = f"/home2/xeweng/forRemote/downstream/_GO/data/hhm_output/{dataset}/dssp"
    hhm_path = f"/home2/xeweng/forRemote/downstream/_GO/data/hhm_output/{dataset}/hhm"
    pssm_path = f'/home2/xeweng/forRemote/downstream/_GO/big_data/data/pssm/{dataset}/pssm'  # 混合特征添加：GO数据Pssm路径

    dssp_path = args.dssp_dir if args.dssp_dir != "" else dssp_path
    hhm_path = args.hhm_dir if args.hhm_dir != "" else hhm_path
    pssm_path = args.pssm_dir if args.pssm_dir != "" else pssm_path

    # 生成DSSP
    # dssppath = os.path.join(output_path, 'dssp')
    # generate_dssp(input_path, dssp_path)
    # import generate_dssp
    # generate_dssp.main(input_path, dssp_path, name_list=name_list)

    # 生成hhm
    # import getFastaAndGenerateHHM

    print("-----generate hhm start-----")
    # getFastaAndGenerateHHM.main(dssp_path, output_path)  # 由函数生成子文件夹
    print("-----generate hhm done-----")

    # 切分词，根据分词获取特征， 并将所有蛋白质的特征合并在一起
    print("----- cut protein and get raw features start-----")
    import getProteinFeature
    feature_path =getProteinFeature.main(output_path, dssp_path=dssp_path, hhm_path=hhm_path, pssm_path=pssm_path, name_list=prots)  #, namelist=prots
    # feature_path = os.path.join(output_path, 'feature')
    print("----- cut protein and get raw features done-----")
    print(feature_path)

