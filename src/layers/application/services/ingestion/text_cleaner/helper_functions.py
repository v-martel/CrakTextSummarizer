from math import ceil


def get_max_length(list_of_processed_headlines: [str]) -> int:
    return max([len(headline.split(" ")) for headline in list_of_processed_headlines])


def get_recommended_length(list_of_processed_headlines: [str]) -> int:
    return ceil(get_max_length(list_of_processed_headlines))


def add_start_end_token(list_of_outputs: list) -> list:
    return list(map(lambda x: "sostok " + x + " eostok", list_of_outputs))
