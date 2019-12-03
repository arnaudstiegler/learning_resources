# TF-IDF

Stands for Term-Frequency - Inverse Document Frequency.

##### Idea:
Instead of doing a base bag-of-words approach (which simply counts each word), 
tf-idf is a weighted count in the sense that rare words will be weighted more than words that occur often. For instance, if two words occur the same amount of time in a document, but one is rare among the corpus and not the other, then the rare word will have a higher count than the frequent word.

##### How it works:
