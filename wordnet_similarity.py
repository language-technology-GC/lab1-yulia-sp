#!/usr/bin/env python

import nltk
from nltk.corpus import wordnet, wordnet_ic
from scipy import stats

brown_ic = wordnet_ic.ic('ic-brown.dat')
PATH = "data/ws353.tsv"

"""

For synset similarity, I decided to find synsets with the highest similarity for each word pair.
The difficult part was figuring out what is required for computing each type of similarity.
For instance, lch_similarity, res_similarity, jcn_similarity, and lin_similarity require the words
to have the same POS tags, and the similarity methods that use Information Context have an 
additional constraint due to certain POS tags ('s', 'a') not present in the IC.

RESULTS:
'jiang_conrath_similarity': 0.5954,
'leacock_chodorow_similarity': 0.615,
'lin_similarity': 0.6089,
'path_similarity': 0.5971,
'resnik_similarity': 0.6376,
'wu_palmer_similarity': 0.6389,
 coverage: 100.0

"""


def path_similarities(filename: str) -> tuple:
    """
    Compute the Spearman's rho correlation between 6 methods of computing word similarity using WordNet:
        path similarity,
        Leacock-Chodorow similarity,
        Wu-Palmer similarity,
        Resnik similarity,
        Jiang-Conrath similarity, and
        Lin similarity
    Return a dictionary of Spearman's rho correlation coefficients for each of 6 methods,
    and the percentage of words supported by the method.
    :param filename: (string) path to the tsv file containing word pairs and human judgements of their similarity
    :return: a tuple of a dictionary and a float
    """
    humans = []
    path_similarity = []
    leacock_chodorow_similarity = []
    wu_palmer_similarity = []
    resnik_similarity = []
    jiang_conrath_similarity = []
    lin_similarity = []
    not_covered = 0
    with open(filename, 'r') as source:
        while True:
            line1 = source.readline()
            if not line1:
                break
            first_word, second_word, human_score = line1.rstrip().lstrip().split('\t')
            humans.append(float(human_score))
            first_synsets = wordnet.synsets(first_word)
            second_synsets = wordnet.synsets(second_word)
            if not first_synsets or not second_synsets:
                not_covered += 1
                continue
            path_sim = 0
            leacock_chodorow = 0
            wu_palmer = 0
            resnik = 0
            jiang_conrath = 0
            lin = 0
            for first_syn in first_synsets:
                for second_syn in second_synsets:
                    if first_syn.path_similarity(second_syn):
                        if round(first_syn.path_similarity(second_syn), 2) > path_sim:
                            path_sim = round(first_syn.path_similarity(second_syn), 2)
                    if first_syn.wup_similarity(second_syn):
                        if round(first_syn.wup_similarity(second_syn), 2) > wu_palmer:
                            wu_palmer = round(first_syn.wup_similarity(second_syn), 2)
                    # lch_similarity, res_similarity, jcn_similarity, and lin_similarity require same pos
                    if first_syn.pos() == second_syn.pos():
                        if first_syn.lch_similarity(second_syn):
                            if round(first_syn.lch_similarity(second_syn), 2) > leacock_chodorow:
                                leacock_chodorow = round(first_syn.lch_similarity(second_syn), 2)
                        # IC requires pos != 's' or 'a'
                        if first_syn.pos() not in ['s', 'a']:
                            if first_syn.res_similarity(second_syn, brown_ic):
                                if round(first_syn.res_similarity(second_syn, brown_ic), 2) > resnik:
                                    resnik = round(first_syn.res_similarity(second_syn, brown_ic), 2)
                            if first_syn.jcn_similarity(second_syn, brown_ic):
                                if round(first_syn.jcn_similarity(second_syn, brown_ic), 2) > jiang_conrath:
                                    jiang_conrath = round(first_syn.jcn_similarity(second_syn, brown_ic), 2)
                            if first_syn.lin_similarity(second_syn, brown_ic):
                                if round(first_syn.lin_similarity(second_syn, brown_ic), 2) > lin:
                                    lin = round(first_syn.lin_similarity(second_syn, brown_ic), 2)
            path_similarity.append(path_sim)
            leacock_chodorow_similarity.append(leacock_chodorow)
            wu_palmer_similarity.append(wu_palmer)
            resnik_similarity.append(resnik)
            jiang_conrath_similarity.append(jiang_conrath)
            lin_similarity.append(lin)
    spearman_rho = {}
    spearman_rho['path_similarity'] = round(stats.spearmanr(humans, path_similarity)[0], 4)
    spearman_rho['leacock_chodorow_similarity'] = round(stats.spearmanr(humans, leacock_chodorow_similarity)[0], 4)
    spearman_rho['wu_palmer_similarity'] = round(stats.spearmanr(humans, wu_palmer_similarity)[0], 4)
    spearman_rho['resnik_similarity'] = round(stats.spearmanr(humans, resnik_similarity)[0], 4)
    spearman_rho['jiang_conrath_similarity'] = round(stats.spearmanr(humans, jiang_conrath_similarity)[0], 4)
    spearman_rho['lin_similarity'] = round(stats.spearmanr(humans, lin_similarity)[0], 4)
    coverage = round(((len(humans) - not_covered)/len(humans))*100, 2)
    return spearman_rho, coverage


if __name__ == '__main__':
    spearman_rho_correlations, coverage = path_similarities("NLP/LANG_TECH/assignments/HW1/data/ws353.tsv")
    print("Spearman's rho correlation coefficients:")
    print(spearman_rho_correlations)
    print("Coverage: ", coverage)

