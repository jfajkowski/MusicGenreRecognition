from gensim import corpora, models
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    main_directory = '/home/fajqa/Documents/Python/MusicGenreRecognition/resources/'
    mm = corpora.MmCorpus(main_directory + 'saved/model.mm')
    id2word = corpora.Dictionary.load(main_directory + 'saved/filtered_dictionary_lower_case.dict')

    lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=40, update_every=1, chunksize=10000, passes=1)
    lda.save(main_directory + 'saved/topics.lda')