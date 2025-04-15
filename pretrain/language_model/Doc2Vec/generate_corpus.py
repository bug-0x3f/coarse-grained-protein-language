from html import parser
import os

def generate_corpus(input_dir, output_filepath):
    sentences = []
 
    for filename in os.listdir(input_dir):
        datafile = os.path.join(input_dir, filename)
        with open(datafile, 'r') as f:
            sentences.append(f.read() + '\n')

    with open(output_filepath, 'w') as f:
        f.writelines(sentences)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sentences_path', '-i', type=str, required=True)
parser.add_argument('--output_file', '-o', type=str, required=True)
args = parser.parse_args()
output_file = args.output_file
print('generating corpus...', output_file)
s_path = args.sentences_path
generate_corpus(s_path, output_file)