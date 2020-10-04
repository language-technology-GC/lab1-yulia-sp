#!/usr/bin/env python

from nltk import word_tokenize
import chardet
import gzip
import logging

"""

Biggest challenge: encoding issue in the gz file!!!

2007 news crawl data with only ASCII:
    Tokenized: 3345329
    Skipped: 42670

"""

# the file was determined to have utf-8 encoding
def _determine_encoding(file_path: str) -> str:
    """
    Sniff out the file's encoding
    :param file_path: string: path to the file
    :return: string: encoding
    """
    with open(file_path, 'rb') as file:
        text = file.read(250000)
        detected = chardet.detect(text)
        file_encoding = detected["encoding"]
    return file_encoding


def tokenize(data_path: str, tokens_output_path: str) -> None:
    """
    Tokenize sentences from a .gz file and saved tokenized sentences into a txt file.
    :param data_path: path to the input gz.file
    :param tokens_output_path: path to the output .txt file
    :return: None
    """
    covered = 0
    not_covered = 0
    file_encoding = _determine_encoding(data_path)
    with open(data_path, 'rb') as source:
        with open(tokens_output_path, 'w') as sink:
            for line in source:
                detection_result = chardet.detect(line)
                encoding = detection_result["encoding"]
                if encoding == 'ascii':
                    text = line.decode(encoding, errors='strict')
                    text = text.rstrip().lstrip().casefold()
                    tokens = word_tokenize(text)
                    print(" ".join(tokens), file=sink)
                    covered += 1
                else:
                    not_covered += 1
    print("Tokenized:", covered)
    print("Skipped:", not_covered)


def two_column_tsv(input_path: str, output_path: str) -> None:
    """
    Extract word pairs from a 3-column tsv file and save them into a 2-column tsv file.
    :param input_path: path to the input tsv file
    :param output_path: path to the output tsv file
    :return: None
    """
    with open(input_path, 'r') as source:
        with open(output_path, 'w') as sink:
            for line in source:
                line = line.lstrip().rstrip().casefold()
                first, second, score = line.split('\t')
                print(first+'\t'+second, file=sink)


if __name__ == '__main__':
    two_column_tsv("data/ws353.tsv", "data/ws353_two.tsv")
    crawl_data_path = "data/news.2007.en.shuffled.deduped"
    tokens_path = 'data/news_2007_en_tokens.txt'
    tokenize(crawl_data_path, tokens_path)
