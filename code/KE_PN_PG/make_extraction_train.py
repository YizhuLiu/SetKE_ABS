"""produce the dataset with (psudo) extraction label"""
import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import curry, compose

from utils import count_data
from metric import compute_rouge_l,compute_rouge_n, compute_rouge_l_summ

import collections

try:
    DATA_DIR = os.environ['DATA']
    print(DATA_DIR)
except KeyError:
    print('please use environment variable to specify data directories')


def _split_words(texts):
    return map(lambda t: t.split(), texts)

def findindex(org, x):
    result = []
    for k,v in enumerate(org): 
        if v == x:
            result.append(k)
    return result

def get_extract_label(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    extracted = []
    scores = []
    if ''.join(art_sents[0]) != '<E>':
       art_sents.insert(0,'<E>')
    indices = list(range(len(art_sents)))
    new_abs_sents = []
    for j in range(len(abs_sents)):
        rouges = list(map(compute_rouge_l(reference=abs_sents[j], mode='r'),
                          art_sents[1:]))
        rouges.insert(0,0)
        ext = max(indices, key=lambda i: rouges[i])
        max_scores = rouges[ext]
        max_exts = collections.Counter(rouges)[max_scores]
        if max_exts != 1:
           max_inds = []
           rouge_f = []
           for idx, score in enumerate(rouges):
               if idx in indices:
                  if score == max_scores:
                     max_inds.append(idx)
                     rouge_f.append(compute_rouge_l_summ(art_sents[idx],abs_sents[j],mode='f'))
           maxrouge = max(list(range(len(max_inds))), key=lambda i: rouge_f[i])
           ext = max_inds[maxrouge]
        if ext == 0:
           ext = 1
        new_art_sents = []
        new_art_sents.append('<E>')
        for i in range(1,len(art_sents)):
            #print(art_sents[i])
            if i < ext:
               new_art_sents.append(art_sents[i]+art_sents[ext])
            elif i > ext:
               new_art_sents.append(art_sents[ext]+art_sents[i])
            else:
               new_art_sents.append(art_sents[ext])
        new_rouges = list(map(compute_rouge_l_summ(refs=abs_sents[j], mode='fr'),
                          new_art_sents[1:]))
        new_rouges_f = [fr[0] for fr in new_rouges]
        new_rouges_r = [fr[1] for fr in new_rouges]
        new_rouges_f.insert(0,0)
        new_rouges_r.insert(0,0)
        new_ext = max(indices, key=lambda i: new_rouges_f[i])
        if new_ext == 0:
           new_ext = 1
        #if ext == new_ext or rouges[ext] >= new_rouges[new_ext]:
        if ext == new_ext or rouges[ext] >= new_rouges_r[new_ext]:
           extracted.append(ext)
           extracted.append(0)
           scores.append(new_rouges_f[ext])
        elif ext < new_ext:
           extracted.append(ext)
           extracted.append(new_ext)
           extracted.append(0)
           scores.append(new_rouges_f[new_ext])
        else:
           extracted.append(new_ext)
           extracted.append(ext)
           extracted.append(0)
           scores.append(new_rouges_f[new_ext])

        #reduce duplication: ab->A bc->B, abc->AB
        new_abs_sents.append(abs_sents[j])
        index = findindex(extracted, 0) 
        #dic = collections.Counter(extracted)
        while(len(index) >= 2):
           #print('in')
           if len(index) == 2:
              overlap = list(set(extracted[:index[-2]]) & set(extracted[index[-2]+1:index[-1]]))
              l = len(overlap)
              if l > 0:
                 new = list(set(extracted[:index[-2]]).union(set(extracted[index[-2]+1:index[-1]])))
                 new.sort()
                 del extracted[:index[-1]+1]
                 extracted = extracted+new
                 extracted.append(0)
                 new_sent = new_abs_sents[-2]+new_abs_sents[-1]
                 del new_abs_sents[-2:]
                 new_abs_sents.append(new_sent)
                 index = findindex(extracted, 0) 
              else:
                 break
           else:
              overlap = list(set(extracted[index[-3]+1:index[-2]]) & set(extracted[index[-2]+1:index[-1]]))
              l = len(overlap)
              if l > 0:
                 new = list(set(extracted[index[-3]+1:index[-2]]).union(set(extracted[index[-2]+1:index[-1]])))
                 new.sort()
                 del extracted[index[-3]+1:index[-1]+1]
                 extracted = extracted+new
                 extracted.append(0)
                 new_sent = new_abs_sents[-2]+new_abs_sents[-1]
                 del new_abs_sents[-2:]
                 new_abs_sents.append(new_sent)
                 index = findindex(extracted, 0) 
              else:
                 break
        if len(index) >=2:
           if len(index) == 2:
              for idx in extracted[:index[-2]]:
                 try:
                    indices.remove(idx)
                 except:
                    continue 
           if len(index) > 2:
              for idx in extracted[index[-3]:index[-2]]:
                 try:
                    indices.remove(idx)
                 except:
                    continue 
        if not indices:
            break
    length = len(new_abs_sents)
    for i in range(length):
        new_abs_sents.insert(i+(i+1), ['<E>'])
    return extracted, scores, new_abs_sents, art_sents

@curry
def process(split, i):
    data_dir = join(DATA_DIR, split)
    with open(join(data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    tokenize = compose(list, _split_words)
    art_sents = tokenize(data['article'])
    abs_sents = tokenize(data['abstract'])
    if art_sents and abs_sents: # some data contains empty article/abstract
        extracted, scores , new_abs_sents, art_sents = get_extract_label(art_sents, abs_sents)
    else:
        extracted, scores, new_abs_sents = [], [], []
    data['extracted'] = extracted
    data['score'] = scores
    data['new_abs_sents'] = [' '.join(s) for s in new_abs_sents]
    data['article'] = [' '.join(s) for s in art_sents]
    print(split, '{}.json'.format(i))
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
    #for split in ['val', 'train','test']:  # no need of extraction label when testing
    for split in ['train']:  # no need of extraction label when testing
        label_mp(split)

if __name__ == '__main__':
    main()
