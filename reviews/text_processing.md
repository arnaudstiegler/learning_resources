# TF-IDF

Stands for Term-Frequency - Inverse Document Frequency.

##### Idea:
Instead of doing a base bag-of-words approach (which simply counts each word), 
tf-idf is a weighted count in the sense that rare words will be weighted more than words that occur often. For instance, if two words occur the same amount of time in a document, but one is rare among the corpus and not the other, 
then the rare word will have a higher count than the frequent word.

The underlying idea is that we care less about words that are very frequent in all documents because they are likely to be stopwords that don't convey significant meaning. On the contrary, rare words are usually significant so their presence should be "upweighted". That way, you convey two informations into 1 term: a) the frequency of the word, b) its importance in the corpus.

##### How it works:
Two parts:
- TF (term frequency): it actually is the normalized term frequency (# of occurences)/(# of words in the document)
- IDF (inverse document frequency): it is actually the log of (# Documents)/ (# Documents that contain the word). The more common the word, the smaller the weight. Note that the weigth cannot be smaller than 1
