from gensim import corpora
import re
from tqdm import tqdm
import os


def load_dictionary():
    dictionary = corpora.Dictionary().load(main_directory +
                                           "saved/filtered_dictionary_lower_case.dict")

    return dictionary


def format_text(text):
    text = re.sub("\W+|[_]+|\d+", " ", text)
    return text.lower().split()


def get_lyrics_paths():
    directory = main_directory + "paths/"

    with open(directory + language + '.txt') as file:
        paths = file.readlines()

    return paths


if __name__ == '__main__':
    main_directory = '/home/fajqa/Documents/Python/MusicGenreRecognition/resources/'
    language = 'en'
    dictionary = load_dictionary()
    corpus = []

    paths = get_lyrics_paths()

    bar_wrapper = tqdm(paths)
    bar_wrapper.set_description("Creating vector representations.")

    for path in bar_wrapper:
        text = open(path.split("\n")[0]).read()
        corpus.append(dictionary.doc2bow(format_text(text)))

    corpora.MmCorpus.serialize(main_directory + 'saved/model.mm', corpus)
