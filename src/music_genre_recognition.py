import gensim


class Song(object):
    def __init__(self, echo_nest_id, lyrics):
        self.echo_nest_id = echo_nest_id
        self.lyrics = lyrics
        self.genre = None

if __name__ == '__main__':
    print("DONE!")
