import sys
from collections import defaultdict
import math
import random
import os
import os.path
from collections import Counter
"""
COMS W4705 - Natural Language Processing - Fall 2020 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    if len(sequence) == 0:
        return sequence

    ngrams = []
    sequence.append('STOP')
    sequence_full = ['START' for i in range(n-1)]
    sequence_full.extend(sequence)
    for idx, word in enumerate(sequence_full):
        gram = tuple(sequence_full[idx:idx+n])
        ngrams.append(gram)
        if gram[n-1] == 'STOP':
            return ngrams
    
    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        # add denominator to calculate unigram probabilities
        values = sum(self.unigramcounts.values())
        # remove START token counts in unigram occurrences
        self.denominator = values - self.unigramcounts[('START',)] 


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int) 

        ##Your code here

        count = 0
        for sent in corpus:
            count += 1
            uni = get_ngrams(sent, 1)
            for one in uni:
                self.unigramcounts[one] += 1

            bi = get_ngrams(sent, 2)
            for two in bi:
                self.bigramcounts[two] += 1

            tri = get_ngrams(sent, 3)
            for three in tri:
                self.trigramcounts[three] += 1

        self.unigramcounts[('START',)] = count
        self.bigramcounts[('START', 'START')] = count
        return 

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        # if unseen bigram
        if self.bigramcounts[trigram[:2]] == 0:
            # approximate trigram (t1, t2, t3) prob with unigram prob of t3 
            # 0 if unigram not seen
            return self.unigramcounts[trigram[2:]]/self.denominator
        else:
            return  self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # if unseen unigram, return 0
        return self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return self.unigramcounts[unigram]/self.denominator

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        return lambda1*self.raw_trigram_probability(trigram)+lambda2*self.raw_bigram_probability(trigram[1:])+lambda3*self.raw_unigram_probability(trigram[2:])
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        logprob = 0.0
        for tri in trigrams:

            logprob += math.log2(self.smoothed_trigram_probability(tri))
        return logprob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        l = 0.0
        M = 0.0
        for sent in corpus:
            l += self.sentence_logprob(sent)
            # add stop occurrences
            M += len(sent) + 1
        l = l/M
        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            total += 1
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp < pp2:
                correct += 1
    
        for f in os.listdir(testdir2):
            total += 1
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp < pp1:
                correct += 1
            # .. 
        
        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    #dev_corpus = corpus_reader(sys.argv[1], model.lexicon)
    #pp = model.perplexity(dev_corpus)
    #print(pp)


    # Essay scoring experiment: 
    #acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt', 'hw1_data/ets_toefl_data/train_low.txt', "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
    #print(acc)

