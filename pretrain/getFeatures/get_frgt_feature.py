#!/usr/bin/env python

import argparse, os
from multiprocessing import Pool
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inputpath', '-i', type=str, default="", required=False, help="a path of input pdb files")
    parser.add_argument('--outputpath', '-o', type=str, default="", required=False,
                        help="a path for saving related output")
    parser.add_argument('-dd', '--dssp_dir', default="", help='dssp文件夹路径')
    parser.add_argument('-pd', '--pssm_dir', default="", help='pssm文件夹路径')
    parser.add_argument('-hhd', '--hhm_dir', default="", help='hhm文件夹路径')
    parser.add_argument('--list', '-l', type=str, default="", required=False, help="a list for select specific pdbs in input_path")

    args = parser.parse_args()


    input_path = args.inputpath
    output_path = os.path.dirname(input_path)
    if input_path.endswith('/'): 
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


    dssp_path = args.dssp_dir
    hhm_path = args.hhm_dir 
    pssm_path = args.pssm_dir

    print("----- cut protein and get raw features start-----")
    import extr_feature
    feature_path = extr_feature.main(output_path, dssp_path=dssp_path, hhm_path=hhm_path, pssm_path=pssm_path, name_list=prots)  
    print("----- cut protein and get raw features done-----")
    print(feature_path)

