import numpy as np
from numpy.lib.function_base import delete
import pandas as pd
import math

tf = lambda wordFreq, maxFreq: wordFreq / maxFreq
class Vector:
    def __init__(self, indices=[], values=[]):
        self.tf_idf = {}
        self.words = {}
        self.maxFrequency = 0
        
    def cosine(self, other):
        pass
    
    def okapi(self, other):
        pass
    
    def addWord(self, word):
        if self.words.get(word) is None:
            self.words[word] = 0
        self.words[word] += 1
        self.maxFrequency = max(self.words[word], self.maxFrequency)
        
    def calcTfIdf(self, n, docFrequencies):
        entries = self.words.items()
        for word, freq in entries:
            if docFrequencies[word] == 1:
                del docFrequencies[word]
            else:
                self.tf_idf[word] = tf(freq, self.maxFrequency) * math.log2(n / docFrequencies[word])      
    
    def removeWord(self, word):
        if word not in self.words:
            return
        del self.words[word]
        self.total -= 1
        
    def outputSparseVector(self):
        values = list(self.tf_idf.values())
        words = list(self.tf_idf.keys())
        return f'{",".join(words)}|{",".join([str(v) for v in values])}'