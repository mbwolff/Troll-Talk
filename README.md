# Troll Talk

## A contribution to [NaNoGenMo 2018](https://github.com/NaNoGenMo/2018)

Earlier this summer [FiveThirtyEight](https://fivethirtyeight.com/features/why-were-sharing-3-million-russian-troll-tweets/) shared a corpus of [nearly three million tweets](https://github.com/fivethirtyeight/russian-troll-tweets/) associated with accounts linked to Russia's Internet Research Agency. The evidence suggests these tweets were part of a campaign to influence the 2016 US election. What was communicated, and how do we make sense of it?

One possibility is to simulate a conversation among the trolls using a word embedding model and tf-idf transforms.

* Build an embedding model of all the words in the Russian troll tweets corpus. This will enable the use of  [Gensim's Word2Vec module](https://radimrehurek.com/gensim/models/word2vec.html), specifically the [most_similar](https://rare-technologies.com/word2vec-tutorial/#using_the_model) function which can generate analogies for each word in a given text with a pair of pre-selected words (such as _liberal_ and _conservative_).
* Transform the corpus of tweets into a [tf-idf matrix](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting).
* Implement the following algorithm until 50,000 words have printed, beginning with a randomly selected tweet.
  * Print the tweet.
  * Remove the tf-idf vector for the tweet from the matrix (this avoids repetition).
  * Replace each word in the tweet by analogy with the word pair and the embedding model.
  * Print the modified tweet.
  * Transform the modified tweet as a tf-idf vector based on the structure of the matrix.
  * Select the tweet for which the vector in the matrix is most similar to the vector of the modified tweet (using cosine similarity).
  * Repeat.
