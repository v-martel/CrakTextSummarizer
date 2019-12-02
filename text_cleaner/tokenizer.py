from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenize_data(headline_article_dict: dict, max_headline_len: int, max_article_len: int) -> dict:
    article_tokenizer = Tokenizer()
    headline_tokenizer = Tokenizer()

    articles_training, articles_validation, headlines_training, headlines_validations = split_training_validation(
        headline_article_dict
    )
    article_tokenizer.fit_on_texts(articles_training)
    headline_tokenizer.fit_on_texts(headlines_training)

    articles_training = tokenize_article(article_tokenizer, max_article_len, articles_training)
    articles_validation = tokenize_article(article_tokenizer, max_article_len, articles_validation)
    articles_voc_size = len(article_tokenizer.word_index) + 1

    headlines_training = tokenize_headline(headline_tokenizer, max_headline_len, headlines_training)
    headlines_validations = tokenize_headline(headline_tokenizer, max_headline_len, headlines_validations)
    headlines_voc_size = len(headline_tokenizer.word_index) + 1

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


def split_training_validation(headline_article_dict: dict) -> ([str], [str], [str], [str]):
    articles_training, articles_validation, headlines_training, headlines_validation = train_test_split(
        headline_article_dict["articles"],
        headline_article_dict["headlines"],
        test_size=0.1,
        random_state=0,
        shuffle=True)

    return articles_training, articles_validation, headlines_training, headlines_validation
