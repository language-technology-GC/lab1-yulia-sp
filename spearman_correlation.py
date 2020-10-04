#!/usr/bin/env python

from scipy import stats


def spearman_rho(annotated_path: str, stats_path: str) -> tuple:
    """
    Count Spearman's rho correlation coefficient for two data sets
    that contain word similarity statistics.
    Print the rho coefficient and the percentage of the words covered.
    :param annotated_path: string: path to the TSV file containing human
        judgments
    :param stats_path: string: path to the TSV file containing
        similarity statistics determined programmatically
    :return: tuple of floats: the rho coefficient and the coverage
    """
    stats_dict = {}
    with open(stats_path, 'r') as statistic:
        for line in statistic:
            if line == '\n':
                continue
            line = line.rstrip().lstrip().casefold()
            first, second, score = line.split('\t')
            pair = (first, second)
            stats_dict[pair] = round(float(score), 2)
    human_judgements = []
    stat_results = []
    covered = 0
    not_covered = 0
    with open(annotated_path, 'r') as judgements:
        for string in judgements:
            string = string.rstrip().lstrip().casefold()
            one, two, similarity = string.split('\t')
            if (one, two) in stats_dict:
                human_judgements.append(similarity)
                stat_results.append(stats_dict[(one, two)])
                covered += 1
            elif (two, one) in stats_dict:
                human_judgements.append(similarity)
                stat_results.append(stats_dict[(two, one)])
                covered += 1
            else:
                not_covered += 1
    rho = round(stats.spearmanr(human_judgements, stat_results)[0], 4)
    print("Spearman's rho:", rho)
    coverage = round((covered / (covered + not_covered))*100, 2)
    print(coverage, '%')
    return rho, coverage


if __name__ == "__main__":
    judgements_path = "data/ws353.tsv"
    stats_path = "data/2007_news_ppmi_05.tsv"
    spearman_rho(judgements_path, stats_path)
