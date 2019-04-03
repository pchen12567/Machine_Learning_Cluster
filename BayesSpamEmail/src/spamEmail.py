import jieba
import os


class SpamEmailBayes:

    # Get Chinese stop words
    def get_stop_words(self):
        stop_list = []
        for line in open('../data/chinese_stop_words.txt'):
            stop_list.append(line[0])
        return stop_list

    # Get the words list
    def get_word_list(self, content, wordsList, stopList):
        # Save the word segmentation results to res_list
        res_list = list(jieba.cut(content))
        for i in res_list:
            if i not in stopList and i.strip() != '' and i is not None:
                if i not in wordsList:
                    wordsList.append(i)

    # Add the word to the words list and account the number
    def addToDict(self, wordsList, wordsDict):
        for word in wordsList:
            if word in wordsDict.keys():
                wordsDict[word] += 1
            else:
                wordsDict.setdefault(word, 1)

    def get_file_list(self, filePath):
        filenames = os.listdir(filePath)
        return filenames

    # Get the top 15 words according to the P(s|w) in each email.
    def get_test_words(self, testDict, spamDict, normDict, normFileLen, spamFileLen):
        word_prob_list = {}
        for word, num in testDict.items():

            if word in spamDict.keys() and word in normDict.keys():
                prob_word_spam = spamDict[word] / spamFileLen
                prob_word_normal = normDict[word] / normFileLen
                prob_spam_word = prob_word_spam / (prob_word_spam + prob_word_normal)
                word_prob_list.setdefault(word, prob_spam_word)

            if word in spamDict.keys() and word not in normDict.keys():
                prob_word_spam = spamDict[word] / spamFileLen
                prob_word_normal = 0.01
                prob_spam_word = prob_word_spam / (prob_word_spam + prob_word_normal)
                word_prob_list.setdefault(word, prob_spam_word)

            if word not in spamDict.keys() and word in normDict.keys():
                prob_word_spam = 0.01
                prob_word_normal = normDict[word] / normFileLen
                prob_spam_word = prob_word_spam / (prob_word_spam + prob_word_normal)
                word_prob_list.setdefault(word, prob_spam_word)

            if word not in spamDict.keys() and word not in normDict.keys():
                word_prob_list.setdefault(word, 0.49)

        word_prob_list = sorted(word_prob_list.items(), key=lambda d: d[1], reverse=True)[0:15]
        return word_prob_list

    # Compute Bayes probability
    def calBayes(self, word_prob_list):
        prob_spam_word = 1
        prob_normal_word = 1

        for item in word_prob_list:
            prob_spam_word *= item[1]
            prob_normal_word *= (1 - item[1])

        p = prob_spam_word / (prob_spam_word + prob_normal_word)
        return p

    def calAccuracy(self, testResult):
        right_count = 0
        error_count = 0
        for name, cat in testResult.items():
            if (int(name) < 1000 and cat == 0) or (int(name) > 1000 and cat == 1):
                right_count += 1
            else:
                error_count += 1
        return right_count / (right_count + error_count)
