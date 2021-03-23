# -*- coding:utf-8 -*-


import numpy as np
from collections import defaultdict


def get_ngrams(text, n):
    """Calculates n-grams.
    Args:
        n: which n-grams to calculate
        text: An array of tokens, each token must be hashable
    Returns:
        A dict storing n-grams count
    """
    assert n > 0
    ngram_dict = defaultdict(int)
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_dict[tuple(text[i:i + n])] += 1
    return ngram_dict


def get_skip_bigrams(text, N=4):
    """Calculates skip 2-grams.
    Args:
        N: Maximum number of unigrams skipped in skip-bigrams
        text: An array of tokens
    Returns:
        A dict storing skip 2-grams count
    """
    assert N >= 0
    ngram_dict = defaultdict(int)
    text_length = len(text)
    max_i = text_length - 2
    for i in range(max_i + 1):
        max_j = min(text_length - 1, i + N + 1)
        for j in range(i + 1, max_j + 1):
            ngram_dict[(text[i], text[j])] += 1
    return ngram_dict


def get_ngrams_multisents(sentences, n):
    """Calculates n-grams for multiple sentences.
    """
    ngram_dict = defaultdict(int)
    for sent in sentences:
        d = get_ngrams(sent, n)
        for k, v in d.items():
            ngram_dict[k] += v
    return ngram_dict


def get_skip_bigrams_multisents(sentences, N=4):
    """Calculates skip-bigrams for multiple sentences.
    """
    ngram_dict = defaultdict(int)
    for sent in sentences:
        d = get_skip_bigrams(sent, N)
        for k, v in d.items():
            ngram_dict[k] += v
    return ngram_dict


def get_top_ngrams(ngram_dict):
    """Calculates most frequent ngrams/skip-bigrams for multiple sentences.
    Args:
        ngram_dict: A dict storing n-grams/skip-bigrams count
    Returns:
        A list storing most frequent n-grams/skip-bigrams (each n-gram/skip-bigram is represented as a tuple)
    """
    max_freq = max(ngram_dict.values())
    return [k for k, v in ngram_dict.items() if v == max_freq]


def get_topk_ngrams(ngram_dict, k):
    """Calculates top-k frequent ngrams/skip-bigrams for multiple sentences.
    Args:
        ngram_dict: A dict storing n-grams/skip-bigrams count
        k: Number of top-k n-grams/skip-bigrams to return
    Returns:
        A list storing top-k frequent n-grams/skip-bigrams (each n-gram/skip-bigram is represented as a tuple)
    """
    ngrams, counts = zip(*ngram_dict.items())
    topk_indices = np.argpartition(counts, -k)[-k:]
    return sorted([ngrams[i] for i in topk_indices], key=ngram_dict.get, reverse=True)
