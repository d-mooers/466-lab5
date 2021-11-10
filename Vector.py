import numpy as np
import pandas as pd

class Vector:
    def __init__(self, indices=[], values=[]):
        self.indices = indices
        self.values = values
        self.words = {}
        
    def cosine(self, other):
        pass
    
    def okapi(self, other):
        pass
    
    def addWord(self, word):
        if self.words.get(word) is None:
            self.words[word] = 0
        self.words[word] += 1