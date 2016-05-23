from gensim import corpora
from langdetect import detect
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_dictionary():
    dictionary = corpora.Dictionary()
    paths = get_lyrics_paths()

    for path in paths:
        file = open(path)
        dictionary.doc2bow(file.read().split(), allow_update=True)

    return dictionary


def get_lyrics_paths():
    paths = []

    for directory, subdirectories, files in os.walk('/home/fajqa/Documents/Python/MusicGenreRecognition/resources/mxm_msd_lyrics_test'):
        for file in files:
            paths.append(os.path.join(directory, file))

    return paths


# collect statistics about all tokens
dictionary = load_dictionary()

# remove stop words and words that appear only once
#stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
#            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
dictionary.filter_tokens(once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)
