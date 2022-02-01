############################################################
# CMPSC 442: Homework 6
############################################################

student_name = "Dmitri Gordienko"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import os
import math


############################################################
# Section 1: Hidden Markov Models
############################################################


def load_corpus(path):
    corpus = []
    with open(path) as file:
        lines = file.readlines()
    for line in lines:
        tuplesline = line.split()
        tuples = []
        for onetupleline in tuplesline:
            onetuple = (onetupleline.split('=')[0], onetupleline.split('=')[1])
            tuples.append(onetuple)
        corpus.append(tuples)
    return corpus


def tag_counter_in_corpus(corpus):
    tag_counters = {}
    for tuples in corpus:
        for (word, tag) in tuples:
            try:
                tag_counters[tag] += 1
            except:
                tag_counters[tag] = 1

    return tag_counters


def tag_counter_per_tag(corpus):
    tag_counters = {}
    prev_tag = None
    for tuples in corpus:
        for (word, tag) in tuples:
            if prev_tag is not None:
                try:
                    try:
                        tag_counters[tag][prev_tag] += 1
                    except:
                        tag_counters[tag][prev_tag] = 1
                except:
                    tag_counters[tag] = {prev_tag: 1}
            prev_tag = tag

    return tag_counters


def words_counter_per_tag(corpus):
    tag_counters = {}
    for tuples in corpus:
        for (word, tag) in tuples:
            try:
                try:
                    tag_counters[tag][word] += 1
                except:
                    tag_counters[tag][word] = 1
            except:
                tag_counters[tag] = {word: 1}

    return tag_counters


def tag_counter_per_word(corpus):
    tag_counters = {}
    for tuples in corpus:
        for (word, tag) in tuples:
            try:
                try:
                    tag_counters[word][tag] += 1
                except:
                    tag_counters[word][tag] = 1
            except:
                tag_counters[word] = {tag: 1}
    return tag_counters


class Tagger(object):

    def __init__(self, sentences):
        self.initial_tag_probabilities = tag_counter_in_corpus(sentences)
        self.transition_probabilities = tag_counter_per_tag(sentences)
        self.emission_probabilities = words_counter_per_tag(sentences)
        self.tag_probabilities = tag_counter_per_word(sentences)

    def most_probable_tags(self, tokens):
        tags = []
        tag_probabilities = {}
        for i in range(0, len(tokens)):
            for tag in self.initial_tag_probabilities.keys():
                word = tokens[i]
                value = math.log(self.word_prob_by_token(self.emission_probabilities[tag], word))
                try:
                    try:
                        tag_probabilities[word][tag] += value
                    except:
                        tag_probabilities[word][tag] = value
                except:
                    tag_probabilities[word] = {tag: value}

        for t in tokens:
            tag_probs = tag_probabilities[t]
            sorted_tag = sorted(tag_probs.items(), key=lambda x: x[1], reverse=True)
            tags.append(sorted_tag[0][0])
        return tags

    def word_prob_by_token(self, dic, key):
        if key in dic:
            return dic[key]
        else:
            return 1e-5

    def viterbi_tags(self, tokens):
        tags = []
        tag_probabilities = {}
        prev_tag = None
        for i in range(len(tokens)):
            for tag in self.initial_tag_probabilities.keys():
                word = tokens[i]
                if prev_tag is None:
                    init_term = math.log(self.initial_tag_probabilities[tag])
                    prev_value = init_term + math.log(self.word_prob_by_token(self.emission_probabilities[tag], word))
                    value = prev_value
                else:
                    trans_prob = math.log(self.word_prob_by_token(self.transition_probabilities[prev_tag], tag))
                    emiss_prob = math.log(self.word_prob_by_token(self.emission_probabilities[tag], word))
                    value = prev_value + trans_prob + emiss_prob
                try:
                    try:
                        tag_probabilities[word][tag] += value
                    except:
                        tag_probabilities[word][tag] = value
                except:
                    tag_probabilities[word] = {tag: value}
                prev_tag = tag
        tag_prob = dict()
        for t in tokens:
            tag_prob = tag_probabilities[t]
            sorted_tag = sorted(tag_prob.items(), key=lambda x: x[1], reverse=True)
            tags.append(sorted_tag[0][0])
        return tags

############################################################
# Section 2: Feedback
############################################################

feedback_question_1 = """
10 hours
"""

feedback_question_2 = """
The viterbi_tags() is tricky to match the outputs.
"""

feedback_question_3 = """
Liked: The idea of building algorithms that do decision making on POS.
Change: Minor thing but brown-corpus.txt(the given file) is not the same as brown_corpus.txt
(used in the pdf file for the test cases), this caused a lot of confusion and wasted time.
"""
