from text_cleaner import get_all_dir_path, get_all_articles_in_dir
from text_cleaner.helper_functions import get_recommended_length
from text_cleaner.tokenizer import tokenize_data
from headline_generator_lstm.headlinesmodel import HeadlinesModel
import warnings

warnings.filterwarnings("ignore")


def print_training_summaries(article_training, headline_training, model):
    for i in range(50):
        print("healine training is : ", headline_training[i])
        print_training_summary(
            model.sequence_to_text(article_training[i]),
            model.decode_sequence(article_training[i].reshape(1, model.max_article_len)),
            model.sequence_to_summary(headline_training[i]))


def print_training_summary(article, predicted_headline, headline):
    print("Article: ", article)
    print("Original headline: ", headline)
    print("Predicted headline: ", predicted_headline, "\n")


def test_training() -> int:
    # headlines_articles_dict = get_all_articles_in_dir("./data/bbc/business")
    headlines_articles_dict = get_all_articles_in_dir("./data/bbc/entertainment")

    max_headline_length = get_recommended_length(headlines_articles_dict["headlines"])
    max_article_length = get_recommended_length(headlines_articles_dict["articles"])

    tokenized_data = tokenize_data(headlines_articles_dict, max_headline_length, max_article_length)
    headlines_training, headlines_validation, headlines_voc_size, headline_index_word, headline_word_index = \
        tokenized_data["headlines"]
    articles_training, articles_validation, articles_voc_size, article_index_word, article_word_index = \
        tokenized_data["articles"]

    for i, j in zip(headlines_articles_dict["headlines"], headlines_articles_dict["articles"]):
        print("headline: ", i)  # , " --- article: ", j)

    print("tokenized data for headlines is: ", tokenized_data["headlines"])
    print("tokenized data for articles is: ", tokenized_data["articles"])

    print("headlines_voc_size = ", headlines_voc_size)
    print("articles_voc_size = ", articles_voc_size)

    headlines_model = HeadlinesModel(
        max_article_len=max_article_length,
        max_headline_len=max_headline_length,

        articles_voc_size=articles_voc_size,
        headlines_voc_size=headlines_voc_size,

        articles_training=articles_training,
        headlines_training=headlines_training,

        articles_validation=articles_validation,
        headlines_validation=headlines_validation,

        article_index_word=article_index_word,
        headline_index_word=headline_index_word,

        article_word_index=article_word_index,
        headline_word_index=headline_word_index)

    print_training_summaries(articles_training, headlines_training, headlines_model)

    return 1
