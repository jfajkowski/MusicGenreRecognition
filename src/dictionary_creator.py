from gensim import corpora
import logging
from tqdm import tqdm
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_dictionary():
    dictionary = corpora.Dictionary()
    paths = get_lyrics_paths()

    bar_wrapper = tqdm(paths)
    bar_wrapper.set_description("Creating dictionary")

    for path in bar_wrapper:
        path = path.split("\n")[0]
        file = open(path)
        lyrics = file.read()
        dictionary.doc2bow(format_text(lyrics), allow_update=True)

    return dictionary


def format_text(text):
    text = re.sub("[^'\w]+|", " ", text)
    return text.lower().split()


def get_lyrics_paths():
    directory = main_directory + "paths/"

    with open(directory + language + '.txt') as file:
        paths = file.readlines()

    return paths


def load_stopwords():
    directory = main_directory + "stopwords/"

    with open(directory + language + '.txt') as file:
        stoplist = file.readlines()

    return stoplist


def filter_dictionary(dictionary):
    stoplist = load_stopwords()

    stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed

if __name__ == '__main__':
    main_directory = '/home/fajqa/Documents/Python/MusicGenreRecognition/resources/'
    language = 'en'

    dictionary = load_dictionary()
    dictionary.save(main_directory + "dictionary_lower_case")
    dictionary.save_as_text(main_directory + "dictionary_lower_case.txt")
    filter_dictionary(dictionary)
    dictionary.save(main_directory + "filtered_dictionary_lower_case")
    dictionary.save_as_text(main_directory + "filtered_dictionary_lower_case.txt")