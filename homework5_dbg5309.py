############################################################
# CMPSC442: Homework 5
############################################################

student_name = "Dmitri Gordienko"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.

import email
import collections
import math
import os

############################################################
# Section 1: Spam Filter
############################################################


def load_tokens(email_path):

    # open the file for reading
    with open(email_path, encoding='utf-8') as file_obj:
        message = email.message_from_file(file_obj)

    # collect the tokens in a list
    tokens = []
    for line in email.iterators.body_line_iterator(message):
        words = line.split()
        for word in words:
            tokens.append(word)

    return tokens


def word_counter_in_emails(email_paths):

    word_counters = dict()

    for email_path in email_paths:
        tokens = load_tokens(email_path)
        for token in tokens:
            if token in word_counters:
               word_counters[token] += 1
            else:
               word_counters[token] = 1

    return word_counters


def log_probs(email_paths, smoothing):

    probs = dict()
    sigma = 0

    # calculate count(w) and sigma
    word_counters = word_counter_in_emails(email_paths)
    for key in word_counters:
        sigma += word_counters[key]

    # log probabilities for each token
    for key in word_counters:
        probs[key] = math.log(word_counters[key] + smoothing) - math.log(sigma + smoothing * (len(word_counters) + 1))

    # log probability for unknown token
    probs["<UNK>"] = math.log(smoothing) - math.log(sigma + smoothing * (len(word_counters) + 1))

    return probs


# create a dictionary with the most indicative value for each word
def most_indicative_value(probs, other_prob):
    most_indicative_probs = dict()
    for key, value in probs.items():
        if key in other_prob:
            d = math.exp(1) ** probs[key] + math.exp(1) ** other_prob[key]
        else:
            d = math.exp(1) ** probs[key] + math.exp(1) ** other_prob["<UNK>"]
        most_indicative_probs[key] = value - math.log(d)
    return most_indicative_probs


# get n keys from sorted dictionaries
def most_indicative_words(sorted_probs, n):
    index = 0
    most_indications = []
    for key, value in sorted_probs:
        most_indications.append(key)
        index += 1
        if index == n:
            return most_indications


class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):
        # collect list of file names
        spam_file_list = os.listdir(spam_dir)
        # create full path name
        spam_paths = [spam_dir + "/" + name for name in spam_file_list]
        # store for later use
        self.log_probs_spam = log_probs(spam_paths, smoothing)

        ham_file_list = os.listdir(ham_dir)
        ham_paths = [ham_dir + "/" + name for name in ham_file_list]
        self.log_probs_ham = log_probs(ham_paths, smoothing)

    def is_spam(self, email_path):
        spam_propability = 0
        ham_propability = 0

        tokens = load_tokens(email_path)
        for token in tokens:
            if token in self.log_probs_spam:
                spam_propability += self.log_probs_spam[token]
            else:
                spam_propability += self.log_probs_spam["<UNK>"]

            if token in self.log_probs_ham:
                ham_propability += self.log_probs_ham[token]
            else:
                ham_propability += self.log_probs_ham["<UNK>"]

        return spam_propability > ham_propability

    def most_indicative_spam(self, n):
        most_indicative_spam_prob = most_indicative_value(self.log_probs_spam, self.log_probs_ham)
        sorted_probs_spam = sorted(most_indicative_spam_prob.items(), key=lambda x: x[1], reverse=True)
        return most_indicative_words(sorted_probs_spam, n)

    def most_indicative_ham(self, n):
        most_indicative_ham_prob = most_indicative_value(self.log_probs_ham, self.log_probs_spam)
        sorted_probs_ham = sorted(most_indicative_ham_prob.items(), key=lambda x: x[1], reverse=True)
        return most_indicative_words(sorted_probs_ham, n)

############################################################
# Section 2: Feedback
############################################################

feedback_question_1 = """
8 hours
"""

feedback_question_2 = """
The most indicative spam and ham functions
"""

feedback_question_3 = """
Learning how to classify spam emails.
"""
