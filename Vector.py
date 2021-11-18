import numpy as np
from numpy.lib.function_base import delete
import pandas as pd
import math

tf = lambda wordFreq, maxFreq: wordFreq / maxFreq

def parseVector(line):
    [author, words, tfidf] = [s.strip() for s in line.split("|")]
    words = [s.strip() for s in words.split(",")]
    tfidf = [float(s.strip()) for s in tfidf.split(",")]
    return Vector(author=author, words=words, values=tfidf)

def okapiInner(fij, fiq, dfi, n, dlj, avDl, k1, k2, b):
    left = math.log((n - dfi + 0.5) / (dfi + 0.5))
    middle = ((k1 + 1) * fij) / (k1 * (1 - b + b * dlj / avDl) + fij)
    right = ((k2 + 1) * fiq) / (k2 + fiq)
    return left * middle * right
class Vector:
    def __init__(self, author="", words=[], values=[]):
        self.tf_idf = {word: tfidf for word, tfidf in zip(words, values)}
        self.words = {}
        self.maxFrequency = 0
        self.length = 0
        self.author = author
        
        if len(values) > 0:
            self.length = np.sqrt(np.sum(np.array(values) ** 2))
        
    # other is another Vector
    def cosine(self, other):
        sharedKeys = set(self.tf_idf.keys()).intersection(set(other.tf_idf.keys()))
        return sum([self.tf_idf[key] * other.tf_idf[key] for key in sharedKeys]) / (self.length * other.length)
    
    def okapi(self, other, documentFrequencies, avgDocLength, n, k1=1, k2=100, b=0.75):
        sharedKeys = set(self.tf_idf.keys()).intersection(set(other.tf_idf.keys()))
        origin = np.array
        return sum([okapiInner(self.tf_idf[word], other.tf_idf.get(word, 0), documentFrequencies[word], len(documentFrequencies), len(self.tf_idf), avgDocLength, k1, k2, b) for word in sharedKeys])
    
    def addWord(self, word):
        if self.words.get(word) is None:
            self.words[word] = 0
        self.words[word] += 1
        self.maxFrequency = max(self.words[word], self.maxFrequency)
        
    def calcTf(self, docFrequencies):
        entries = self.words.items()
        for word, freq in entries:
            if docFrequencies[word] == 1:
                del docFrequencies[word]
            else:
                self.tf_idf[word] = tf(freq, self.maxFrequency) 
                self.length += self.tf_idf[word] ** 2
        self.length = math.sqrt(self.length)     
    
    def calcTfIdf(self, n, docFrequencies):
        entries = self.words.items()
        for word, freq in entries:
            if docFrequencies.get(word) is None:
                continue
            if docFrequencies.get(word, 0) == 1 or docFrequencies.get(word, 5000) > 2500:
                del docFrequencies[word]
            else:
                self.tf_idf[word] = tf(freq, self.maxFrequency) * math.log2(n / docFrequencies[word]) 
                self.length += self.tf_idf[word] ** 2
        self.length = math.sqrt(self.length)     
    
    def removeWord(self, word):
        if word not in self.words:
            return
        del self.words[word]
        self.total -= 1
        
    def outputSparseVector(self):
        values = list(self.tf_idf.values())
        words = list(self.tf_idf.keys())
        return f'{",".join(words)}|{",".join([str(v) for v in values])}'