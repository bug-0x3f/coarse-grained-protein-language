
import argparse, os
from multiprocessing import Pool
import numpy as np
from gensim.models import Doc2Vec

def generate_sentence_vector(filepath): 
    sentence_file = os.path.join(sentences_path, filepath.split('/')[-1])
    with open(sentence_file, 'r') as f:
        indexs = f.read().split()
    
    res = doc2vec_model.infer_vector(indexs)
    res = res.reshape((1, len(res)))
    
    prot = filepath.split('/')[-1].split('.')[0]
    vector_file = os.path.join(vector_path, prot)
    
    np.save(vector_file, res)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sentence_path', '-s', type=str, default="", required=True, help="a path of input sentence files")
    parser.add_argument('--outputpath', '-o', type=str, default="", required=False,
                        help="a path for saving vectors file")
    parser.add_argument('--doc2vec_model', '-d', type=str, default="", required=True,
                        help="a doc2vec model path for get vector to protein representation")
  
    args = parser.parse_args()
    output_path = args.outputpath  
    sentences_path = args.sentence_path
    vector_path = args.outputpath

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    doc2vec_model_path = args.doc2vec_model

    doc2vec_model = Doc2Vec.load(doc2vec_model_path)
    files = os.listdir(sentences_path)


    if not os.path.exists(vector_path):
        os.makedirs(vector_path)
    pool = Pool(48)
    params = [os.path.join(sentences_path, file) for file in files]
    pool.map(generate_sentence_vector, params)
    pool.close()
    pool.join()

    print(vector_path)