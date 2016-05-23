from langdetect import detect
from langdetect import PROFILES_DIRECTORY
from tqdm import tqdm
import os


def save_paths(directory):
    sorted_paths = sort_paths_by_lyrics_language()
    os.mkdir(directory)

    bar_wrapper = tqdm(sorted_paths.keys())
    bar_wrapper.set_description("Saving")

    for language in bar_wrapper:
        file = open(directory + language + '.txt', 'w')
        for path in sorted_paths.get(language):
            file.write(path + '\n')


def sort_paths_by_lyrics_language():
    paths = get_lyrics_paths()

    languages = os.listdir(PROFILES_DIRECTORY)
    languages.append('missing')

    sorted_paths = {language: [] for language in languages}

    bar_wrapper = tqdm(paths)
    bar_wrapper.set_description("Detecting language")

    for path in bar_wrapper:
        file = open(path)
        try:
            language = detect(file.read())
            sorted_paths.get(language).append(path)
        except Exception:
            sorted_paths.get('missing').append(path)

    return sorted_paths


def get_lyrics_paths():
    paths = []

    bar_wrapper = tqdm(
            os.walk('/home/fajqa/Documents/Python/MusicGenreRecognition/resources/mxm_msd_lyrics'))
    bar_wrapper.set_description("Collecting paths")

    for directory, subdirectories, files in bar_wrapper:
        for file in files:
            paths.append(os.path.join(directory, file))

    return paths


if __name__ == '__main__':
    save_paths('/home/fajqa/Documents/Python/MusicGenreRecognition/resources/paths/')
