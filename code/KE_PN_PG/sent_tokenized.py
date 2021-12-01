"""produce the dataset with (psudo) extraction label"""
import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import curry, compose

from utils import count_data
from metric import compute_rouge_l

import nltk


try:
    #DATA_DIR = os.environ['DATA']
    DATA_DIR = '../cnn-dailymail/finished_files'
except KeyError:
    print('please use environment variable to specify data directories')


def _split_words(texts):
    return map(lambda t: t.split(), texts)


def sent_tokenize(sents):
    if len(sents):
       text = ' '.join(sents)
       new_sents = nltk.sent_tokenize(text)
       return new_sents
    else:
       return sents

@curry
def process(split, i):
    data_dir = join(DATA_DIR, split)
    with open(join(data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    art_sents = sent_tokenize(data['article'])
    abs_sents = sent_tokenize(data['abstract'])
    data['article'] = art_sents
    data['abstract'] = abs_sents

    with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
        json.dump(data, f, indent=4)

def label_mp(split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(split),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))

def label(split):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data),
              end='')
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        tokenize = compose(list, _split_words)
        art_sents = tokenize(data['article'])
        abs_sents = tokenize(data['abstract'])
        extracted, scores = get_extract_label(art_sents, abs_sents)
        data['extracted'] = extracted
        data['score'] = scores
        with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
            json.dump(data, f, indent=4)
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main():
    for split in ['val', 'train','test']:  # no need of extraction label when testing
        label_mp(split)

if __name__ == '__main__':
    main()
