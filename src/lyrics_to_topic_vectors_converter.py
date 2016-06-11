import gensim
import os
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_paths():
    directory = main_directory + "paths/"

    with open(directory + language + '.txt') as file:
        paths = file.readlines()

    return paths


def create_directory(path):
    directory = path.replace('mxm_msd_lyrics', 'vectors').split('\n')[0]
    if not os.path.exists(directory[0:-22]):
        os.makedirs(directory[0:-22])
    return directory


if __name__ == '__main__':
    main_directory = '/home/fajqa/Documents/Python/MusicGenreRecognition/resources/'
    language = 'en'

    lda = gensim.models.LdaModel.load(main_directory + 'saved/topics.lda')
    corpus = gensim.corpora.MmCorpus(main_directory + 'saved/model.mm')
    corpus_lda = lda[corpus]
    paths = get_paths()

    bar_wrapper = tqdm(range(0, 40))
    bar_wrapper.set_description("Saving vector representations")

    for i in bar_wrapper:
        with open(main_directory + "topics.txt", 'a') as file:
            file.write('Topic ' + str(i) + ': ' + lda.print_topic(i)+'\n')

    #bar_wrapper = tqdm(range(0, len(paths)))
    #bar_wrapper.set_description("Saving vector representations")

    #for i in bar_wrapper:
     #   current_directory = create_directory(paths[i])
      #  with open(current_directory, "w") as file:
       #     file.write('\n'.join('%s %s' % vector_tuple for vector_tuple in corpus_lda[i]))

    #print "DONE!"
