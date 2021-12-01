"""produce the dataset with (psudo) extraction label"""
import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import curry, compose

from utils import count_data


"""Script to run through new cleanse and RAKE logic"""
import os
import RAKE
import tfidf
import textrank
import nltk
nltk.download('averaged_perceptron_tagger')

RAKE_STOPLIST = 'stoplists/SmartStoplist.txt'

try:
    #DATA_DIR = os.environ['DATA']
    DATA_DIR = '../debug'
except KeyError:
    print('please use environment variable to specify data directories')


def _split_words(texts):
    return map(lambda t: t.split(), texts)


def get_keywords(art_sents, abs_sents):
    """ greedily match summary sentences to article sentences"""
    keywords = {} 
    art = ' '.join(art_sents[1:])
    abss = ' '.join(abs_sents)
    #3: RAKE keywords for each doc 
    rake = RAKE.Rake(RAKE_STOPLIST, min_char_length=2, max_words_length=5)
    keywords["rake_art"] = rake.run(art)
    keywords["rake_abs"] = rake.run(abss)

    #4: TF-IDF keywords for processed text
    art_frequencies = {}
    abs_frequencies = {}
    document_count = 1

    keywords["tfidf_freq_art"] = tfidf.get_word_frequencies(art)
    for word in keywords["tfidf_freq_art"]:
            art_frequencies.setdefault(word, 0)
            art_frequencies[word] += 1
    keywords["tfidf_freq_abs"] = tfidf.get_word_frequencies(abss)
    for word in keywords["tfidf_freq_abs"]:
            abs_frequencies.setdefault(word, 0)
            abs_frequencies[word] += 1

    sortby = lambda x: x[1]["score"]
    for word in keywords["tfidf_freq_art"].items():
            word_frequency = word[1]["frequency"]
            docs_with_word = art_frequencies[word[0]]
            word[1]["score"] = tfidf.calculate(word_frequency, document_count, docs_with_word)
    sortby = lambda x: x[1]["score"]
    for word in keywords["tfidf_freq_abs"].items():
            word_frequency = word[1]["frequency"]
            docs_with_word = abs_frequencies[word[0]]
            word[1]["score"] = tfidf.calculate(word_frequency, document_count, docs_with_word)

    keywords["tfidf_art"] = sorted(keywords["tfidf_freq_art"].items(), key=sortby, reverse=True)
    keywords["tfidf_abs"] = sorted(keywords["tfidf_freq_abs"].items(), key=sortby, reverse=True)

    #5. TextRank
    keywords['textrank_art'] = textrank.extractKeyphrases(art)
    keywords['textrank_abs'] = textrank.extractKeyphrases(abss)
    
    return keywords['rake_art'], keywords['rake_abs'],keywords['tfidf_art'], keywords['tfidf_art'], keywords['textrank_art'], keywords['textrank_art']

@curry
def process(split, i):
    data_dir = join(DATA_DIR, split)
    with open(join(data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    art_sents = data['article']
    abs_sents = data['abstract']
    if art_sents and abs_sents: # some data contains empty article/abstract
        rake_art, rake_abs, tfidf_art, tfidf_abs, tr_art, tr_abs = get_keywords(art_sents, abs_sents)
    else:
        rake_art, rake_abs, tfidf_art, tfidf_abs, tr_art, tr_abs = [],[],[],[],[],[]
    data['rake_art'] = rake_art
    data['rake_abs'] = rake_abs
    data['tfidf_art'] = tfidf_art
    data['tfidf_abs'] = tfidf_abs
    data['tr_art'] = tr_art
    data['tr_abs'] = tr_abs
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
    for split in ['test']:  # no need of extraction label when testing
        label_mp(split)

if __name__ == '__main__':
    main()
