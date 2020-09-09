from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenize_data(headline_article_dict: dict, max_headline_len: int, max_article_len: int) -> dict:
    articles_training, articles_validation, headlines_training, headlines_validations = split_training_validation(
        headline_article_dict
    )

    article_tokenizer = Tokenizer()
    headline_tokenizer = Tokenizer()

    article_tokenizer.fit_on_texts(articles_training)
    headline_tokenizer.fit_on_texts(headlines_training)

    article_rare_count, article_total_count, article_rare_frequency, article_total_frequency = handle_rare_words(
        article_tokenizer, 5)
    headline_rare_count, headline_total_count, headline_rare_frequency, headline_total_frequency = handle_rare_words(
        headline_tokenizer, 3)

    # prepare a tokenizer for reviews on training data
    article_tokenizer = Tokenizer(num_words=article_total_count - article_rare_count)
    article_tokenizer.fit_on_texts(articles_training)

    headline_tokenizer = Tokenizer(num_words=headline_total_count - headline_rare_count)
    headline_tokenizer.fit_on_texts(headlines_training)

    # convert text sequences into integer sequences and padding zero upto maximum length
    articles_training = tokenize_article(article_tokenizer, max_article_len, articles_training)
    articles_validation = tokenize_article(article_tokenizer, max_article_len, articles_validation)

    headlines_training = tokenize_headline(headline_tokenizer, max_headline_len, headlines_training)
    headlines_validations = tokenize_headline(headline_tokenizer, max_headline_len, headlines_validations)

    # size of vocabulary
    articles_voc_size = article_tokenizer.num_words + 1
    headlines_voc_size = headline_tokenizer.num_words + 1

    return {
        "headlines": (headlines_training,
                      headlines_validations,
                      headlines_voc_size,
                      headline_tokenizer.index_word,
                      headline_tokenizer.word_index
                      ),
        "articles": (articles_training,
                     articles_validation,
                     articles_voc_size,
                     article_tokenizer.index_word,
                     article_tokenizer.word_index
                     )
    }


def tokenize_headline(headline_tokenizer: Tokenizer, max_headline_len: int, headline_list: [str]):
    new_headline_list = headline_tokenizer.texts_to_sequences(headline_list)
    return pad_sequences(new_headline_list, maxlen=max_headline_len, padding="post")


def tokenize_article(article_tokenizer: Tokenizer, max_article_len: int, article_list: [str]):
    new_article_list = article_tokenizer.texts_to_sequences(article_list)
    return pad_sequences(new_article_list, maxlen=max_article_len, padding="post")


def handle_rare_words(tokenizer: Tokenizer, threshold: int) -> (int, int, int, int):
    count = 0
    total_count = 0
    frequency = 0
    total_frequency = 0

    for key, value in tokenizer.word_counts.items():
        total_count += 1
        total_frequency += value
        if value < threshold:
            count += 1
            frequency += value

    return count, total_count, frequency, total_frequency


def split_training_validation(headline_article_dict: dict) -> ([str], [str], [str], [str]):
    articles_training, articles_validation, headlines_training, headlines_validation = train_test_split(
        headline_article_dict["articles"],
        headline_article_dict["headlines"],
        test_size=0.15,
        random_state=0,
        shuffle=True)

    return articles_training, articles_validation, headlines_training, headlines_validation
