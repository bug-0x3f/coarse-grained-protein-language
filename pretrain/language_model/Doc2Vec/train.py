import os

from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

from datetime import datetime


vector_dim = 600
epochs = 150

def parse():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--corpusfile', '-c', type=str, default="", required=True, help="a path of input pdb files")
    parser.add_argument('--outputpath', '-o', type=str, default="", required=False, help="a path for saving vectors file")
    parser.add_argument('--model_name', '-m', type=str, default="doc2vec.model", required=False, help="name the model file")

    args = parser.parse_args()
    corpusfile = args.corpusfile
    output_path = args.outputpath
    model_name = args.model_name

    return model_name, output_path, corpusfile

model_name, output_path, corpusfile = parse()

if not os.path.exists(output_path):
    os.makedirs(output_path)
with open(corpusfile, 'r') as f:
    sentences = [line.strip().split() for line in f.readlines()]

print('Number of sentences in corpus', len(sentences))
model = Doc2Vec(vector_size=vector_dim, min_count=2
                    , epochs=epochs
                    , seed=42 
                    , alpha=0.025
                    , min_alpha=0.0001
                    , workers=256
                    )

train_corpus = [TaggedDocument(words=sentence, tags=[i]) for i, sentence in enumerate(sentences)]  
model.build_vocab(train_corpus)
st = datetime.now()

if not os.path.exists(output_path):
    os.mkdir(output_path)
print('start traing:', st)
model.train(train_corpus
            , total_examples=model.corpus_count
            , epochs=model.epochs
            )
model.save(os.path.join(output_path, model_name))
print('Train finshed, need time:', datetime.now() - st)