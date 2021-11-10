import nltk
import numpy as np
import pandas as pd
from Vector import Vector
import glob
import os
import argparse
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

sno = nltk.stem.PorterStemmer()

preVectorizeAddWord = lambda vector, total: lambda word: addWord(word, vector, total)

def addWord(word, vector, total):
    word = sno.stem(word.lower())
    if total.get(word) is None:
        total[word] = len(total)
    vector.addWord(word)

def vectorize(directory):
    documentFiles = glob.glob(f'{directory}/*/*/*.txt')
    docs = []
    stopWords = set(stopwords.words('english'))
    totalWords = {}
    for file in documentFiles:
        with open(file, 'r') as fp:
            vector = Vector()
            name = os.path.basename(file.split('.')[0])
            dir = os.path.dirname(file.split('.')[0])
            text = " ".join(fp.readlines())
            # words = np.array(re.findall("[A-Z\-\']{2,}(?![a-z])|[A-Z\-\'][a-z\-\']+(?=[A-Z])|[\'\w\-]+", text.strip()))
            words = np.array(re.split("[\W]+", text))
            words = [word for word in words if word.lower() not in stopWords]
            np.vectorize(preVectorizeAddWord(vector, totalWords))(words)
    print(len(totalWords))
    
def parse():
    parser = argparse.ArgumentParser(description="C45")
    parser.add_argument(
        "directory", type=str, help="directory of 50-50 dataset"
    )

    args = vars(parser.parse_args())
    return args


def main():
    args = parse()
    dataDirectory = args["directory"]
    vectorize(dataDirectory)
    
if __name__ == "__main__": 
    main()