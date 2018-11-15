# Troll Talk

## A contribution to NaNoGenMo 2018

Earlier this summer [FiveThirtyEight](https://fivethirtyeight.com/features/why-were-sharing-3-million-russian-troll-tweets/) shared [nearly three million tweets](https://github.com/fivethirtyeight/russian-troll-tweets/) associated with accounts linked to Russia's Internet Research Agency. The evidence suggests these tweets were part of a campaign to influence the 2016 election. What was communicated, and how do we make sense out of it?

One possibility is to simulate a conversation among the trolls using a word vector space and tf-idf transforms.

* Build a vector space of all the words in Russian troll tweets corpus. This will enable the use of  gensim's Word2Vec module, specifically the most_similar function which can generate analogies for each word in a given text with a pair of pre-selected words (such as *great* and *sad*). For example, the word *Clinton* would be replaced with the word *Cosby*:
```
>>> model.wv.most_similar(positive=[u'great'] + [u'clinton'], negative=[u'sad'], topn=5)
[(u'cosby', 0.5857975482940674), (u'campaign', 0.5846050381660461), (u'presidential', 0.5707162022590637), (u'senate', 0.5666453838348389), (u'final', 0.5653393268585205)]
```
* Transform the corpus of tweets as a tf-idf matrix.
* Implement the following algorithm until 50,000 words have appeared, beginning with a randomly selected tweet.
  1. Print the tweet.
  1. Remove the tf-idf vector for the tweet from the matrix (this avoids repetition).
  1. Replace each word in the tweet by analogy with the word pair and the vector space.
  1. Print the modified tweet.
  1. Transform the modified tweet as a tf-idf vector based on the structure of the matrix.
  1. Select the tweet for which the vector in the matrix is most similar to the vector of the modified tweet (using cosine similarity).
  1. Repeat.
