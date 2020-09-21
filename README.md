MP1: Word similarity
====================

In this lab you will evaluate several methods for computing word similarity
using [WordNet](https://wordnet.princeton.edu/), pointwise mutual information,
and word2vec-style word embedding.

For all three methods, you will compute two statistics:

1.  the *correlation* (the [Spearman
    rho](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
    rank correlation coefficient) with human judgments of similarity , and
2.  the *coverage*, the percentage of word pairs in the database "supported" by
    the method.

The human judgments come from the [WordSimilarity-353 Test
Collection](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/) and
can be found in [`data/ws353.tsv`](data/ws353.tsv). This data set contains 353
word pairs rated for *similarity* on a scale from 0-10, combined across
annotators.

Part 1: WordNet
---------------

Compute the correlation and coverage of WordNet similarity using the six methods
supported by NLTK:

-   *path similarity*,
-   *Leacock-Chodorow similarity*,
-   *Wu-Palmer similarity*,
-   *Resnik similarity*,
-   *Jiang-Conrath similarity*, and
-   *Lin similarity*.

### What to turn in

-   the Spearman correlation coefficient (to 4 decimals) for each of these six
    methods,
-   the number of word pairs "covered" by the method (not all words are present
    in WordNet),
-   any code used, and
-   a brief description of your approach and any problems you ran into.

### Hints

-   There are (at least) two Python libraries for WordNet:
    [NLTK](http://www.nltk.org/) and
    [wordnet](https://github.com/alvations/wordnet). Either way, you will need
    to figure out someway to pick an appropriate synset for each word ("lemma"
    in WordNet terms). The first one returned is supposedly to be the most
    frequent sense (in some corpus). A slightly more sophisticated approach is
    to pick whichever pair of synsets gives you the highest similarity score for
    the pair.
-   [SciPy](https://www.scipy.org/) has an implementation of Spearman
    correlation.

Part 2: Positive pointwise mutual information
---------------------------------------------

Compute the correlation and coverage of *positive pointwise mutual information*
using the provided [`ppmi.py`](ppmi.py) script to compute a PPMI table. You
should:

1.  Download a large English text file from the [WMT news crawl
    collection](http://data.statmt.org/news-crawl/en/).
2.  Tokenize it using the NLTK word tokenizer (the news crawl data is already
    split into sentences).
3.  Run the provided PPMI script over the tokenized data.

### What to turn in

-   the Spearman correlation coefficient (to 4 decimals) for each of these six
    methods,
-   the number of word pairs "covered" by the method,
-   any code (other than the provided script) used,
-   a brief description of your approach and any problems you ran into.

### Hints

-   Computing the PPMI table with the provided script may take as much as an
    hour; this is to be expected as it's processing a lot of text.
-   If you're not sure how to use the provided script, run `./ppmi.py --help`,
    which will print out some useful help information.
-   You will need to create the two-column format expected by
    [`ppmi.py`](ppmi.py)'s `--pairs_path` argument: it will not work if you pass
    it a three-column file.
-   Or if you're totally stumped, write your own PPMI script.
-   The provided script returns the word pairs in lexicographic order, but they
    are not sorted in the original WordSimilarity-353 data.

### Stretch goal

Tune the parameters of the PPMI model by varying the window size, or by
modifying the script so that it supports a flag specifying an add-alpha
smoothing coefficient.

Part 3: Word2Vec similarity
---------------------------

Compute the correlation and coverage of *word2vec*-style word similarity using
[Gensim](https://radimrehurek.com/gensim/index.html) and the provided
[`word2vec.py`](word2vec.py) script.

You should:

1.  Install Gensim (`conda install gensim` if you use Conda;
    `pip install gensim` otherwise).
2.  Run the provided script over the tokenized data from part 2.

### What to turn in

-   the Spearman correlation coefficient (to 4 decimals) for each of these six
    methods,
-   the number of word pairs "covered" by the method,
-   any code (other than the provided script) used,
-   a brief description of your approach and any problems you ran into.

### Hints

-   Computing the word2Vec table with the provided script may take as much as an
    hour; once again, this is to be expected.
-   Read [the docs](https://radimrehurek.com/gensim/models/word2vec.html) if you
    get stuck.
-   If you're not sure how to use the provided script, run
    `./word2vec.py --help`, which will print out some useful help information.
-   You will need to create the two-column format expected by
    \[`word2vec.py`\](word2vec.py\]'s `--pairs_path` argument: it will not work
    if you pass it a three-column file.
-   Or if you're totally stumped, write your own word2vec script.
-   The provided script returns the word pairs in lexicographic order, but they
    are not sorted in the original WordSimilarity-353 data.

### Stretch goal

Tune the parameters of the Word2Vec model by varying the various
[hyperparamters](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec),
adding flags to the command-line.

Alternatively, experiment with alternatives like
[FastText](https://fasttext.cc/) or [GenSim's
clone](https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText),
which also uses character embeddings, or
[GloVe](https://nlp.stanford.edu/projects/glove/), which uses an alternative
model. Install these tools, train your own embeddings, and report the results.

Part 4: Summary
---------------

Briefly summarize your results for the three word similarity techniques.

### What to turn in

A short summary of the three methods' correlation with human judgments and
general coverage.
