# python -m lilab.lstm_bhv_bodylennorm_classify.s3_clippredpkl_append_umap  CLIPPREDPKL_FILE
import pickle
import argparse
from sklearn.neural_network import MLPRegressor


def main(clippredfile):
    clippreddata = pickle.load(open(clippredfile,'rb'))
    embedding = clippreddata['embedding']
    mlp = pickle.load(open('/home/liying_lab/chenxf/ml-project/LILAB-py/lilab/lstm_bhv_bodylennorm_classify/norm_k36_MLP_embedding_to_d2.pkl', 'rb'))['mlp']
    embedding_d2_raw = mlp.predict(embedding)
    embedding_d2 = (embedding_d2_raw + [[6, 6]]) / [[1.4, 1.1]]
    clippreddata['embedding_d2'] = embedding_d2
    pickle.dump(clippreddata, open(clippredfile, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clippredfile', type=str, help='clippredfile')
    args = parser.parse_args()
    main(args.clippredfile)
