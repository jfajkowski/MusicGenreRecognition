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
    text = re.sub("\W+|[_]+|\d+", " ", text)
    return text.lower().split()


def get_lyrics_paths():
    directory = main_directory + "paths/"

    with open(directory + language + '.txt') as file:
        paths = file.readlines()

    return paths


def load_stopwords():
    directory = main_directory + "stopwords/"

    with open(directory + language + '.txt') as file:
        stoplist = file.read().splitlines()

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

    dictionary = corpora.Dictionary.load('/home/fajqa/Documents/Python/MusicGenreRecognition/resources/saved/dictionary_lower_case.dict')#load_dictionary()
    #dictionary.save(main_directory + "dictionary_lower_case.dict")
    #dictionary.save_as_text(main_directory + "dictionary_lower_case.txt")
    filter_dictionary(dictionary)
    dictionary.save(main_directory + "saved/filtered_dictionary_lower_case.dict")
    dictionary.save_as_text(main_directory + "saved/filtered_dictionary_lower_case.txt", sort_by_word=False)