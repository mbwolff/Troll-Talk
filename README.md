# Troll Talk

## A contribution to NaNoGenMo 2018

Earlier this summer [FiveThirtyEight](https://fivethirtyeight.com/features/why-were-sharing-3-million-russian-troll-tweets/) shared [nearly three million tweets](https://github.com/fivethirtyeight/russian-troll-tweets/) associated with accounts linked to Russia's Internet Research Agency. The evidence suggests these tweets were part of a campaign to influence the 2016 election. What was communicated, and how do we make sense out of it?

One possibility is to simulate a conversation among the trolls using a word vector space and tf-idf transforms.

* Build a vector space of all the words in Russian troll tweets corpus. This will enable the use of  gensim's Word2Vec module, specifically the most_similar function which can generate analogies for each word in a given text with a pair of pre-selected words.
* Transform the corpus of tweets as a tf-idf matrix.
* Implement the following algorithm until 50,000 words have appeared, beginning with a randomly selected tweet.
  * Print the tweet.
  * Remove the tf-idf vector for the tweet from the matrix (this avoids repetition).
  * Replace each word in the tweet by analogy with the word pair and the vector space.
  * Print the modified tweet.
  * Transform the modified tweet as a tf-idf vector based on the structure of the matrix.
  * Select the tweet for which the vector in the matrix is most similar to the vector of the modified tweet (using cosine similarity).
  * Repeat.
