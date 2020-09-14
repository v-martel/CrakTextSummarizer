import re

from os import listdir
from os.path import isfile, isdir, join
from data.contractions import contraction_mapping
from nltk.corpus import stopwords

from src.layers.application.services.ingestion.text_cleaner.helper_functions import add_start_end_token
from src.layers.domain.model.digestible import Digestible


def get_all_dir_path(path_to_data: str) -> [str]:
    return [
        "/".join([path_to_data, directory]) for directory in listdir(path_to_data) if
        isdir(join(path_to_data, directory))
    ]


def get_all_articles_in_dir(path_to_dir: str) -> dict:
    stop_words = set(stopwords.words("english"))
    file_list = ["/".join([path_to_dir, file]) for file in listdir(path_to_dir) if isfile(join(path_to_dir, file))]
    articles_dict = {"headlines": [], "articles": []}

    for file_path in file_list:
        current_article = from_file_create_training_case(stop_words, file_path)
        articles_dict["headlines"].append(current_article[0])
        articles_dict["articles"].append(current_article[1])

    articles_dict["headlines"] = add_start_end_token(articles_dict["headlines"])

    return articles_dict


def from_file_create_training_case(stop_words: set, path_to_file: str) -> [str, str]:
    with open(path_to_file) as current_file:
        content = [line.strip() for line in current_file.readlines()]
    return [
        clean_string(stop_words, content[0]),
        clean_string(stop_words, " ".join(content[1:]))
    ]


def clean_Digestible(digestible: Digestible):
    stop_words = set(stopwords.words("english"))

    inputs = list(map(lambda x: clean_string(stop_words, x), digestible.inputs))
    outputs = list(map(lambda x: clean_string(stop_words, x), digestible.outputs))

    return Digestible(inputs, outputs)


def clean_string(stop_words: set, string: str) -> str:
    # this was almost exclusively copied from:
    #   https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/
    output_string = string.lower()
    output_string = re.sub(r'\([^)]*\)', '', output_string)
    output_string = re.sub('"', '', output_string)
    output_string = ' '.join(
        [contraction_mapping[t] if t in contraction_mapping else t for t in output_string.split(" ")]
    )
    output_string = re.sub(r"'s\b", "", output_string)
    output_string = re.sub("[^a-zA-Z]", " ", output_string)
    tokens = [word for word in output_string.split() if word not in stop_words]
    long_words = []

    for word in tokens:
        if len(word) >= 3:
            long_words.append(word)

    return (" ".join(long_words)).strip()
