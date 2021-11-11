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
stopWords = set(stopwords.words('english'))


sno = nltk.stem.PorterStemmer()

preVectorizeAddWord = lambda vector, wordAdded, df: lambda word: addWord(word, vector, wordAdded, df)

def addWord(word, vector, wordAdded, df):
    if word.lower() in stopWords:
        return
    
    word = sno.stem(word.lower())
    updateDocumentFrequency(df, word, wordAdded)
    vector.addWord(word)
    
def updateDocumentFrequency(df, word, wordAdded):
    if df.get(word) is None:
        df[word] = 0
    if word not in wordAdded:
        df[word] += 1
        wordAdded.add(word)

def vectorize(directory, dataOut, truthOut):
    documentFiles = glob.glob(f'{directory}/*/*/*.txt')
    
    vectors = []
    authors = []
    fileNames = []
    documentFrequency  = {}
    allVectors = {}
    sparseOutPut = []
    for file in documentFiles:
        wordCounted = set()
        with open(file, 'r') as fp:
            vector = Vector()
            name = os.path.basename(file.split('.')[0])
            author = os.path.dirname(file.split('.')[0]).split("\\")[2]
            text = " ".join(fp.readlines())
            # words = np.array(re.findall("[A-Z\-\']{2,}(?![a-z])|[A-Z\-\'][a-z\-\']+(?=[A-Z])|[\'\w\-]+", text.strip()))
            words = np.array(re.split("[\W]+", text))
            np.vectorize(preVectorizeAddWord(vector, wordCounted, documentFrequency))(words)
            vectors.append(vector)
            authors.append(author)
            fileNames.append(name)
    for vector in vectors:
        vector.calcTfIdf(len(documentFiles), documentFrequency)
        sparseOutPut.append(vector.outputSparseVector())
    for word in documentFrequency.keys():
        allVectors[word] = [vector.tf_idf.get(word, 0) for vector in vectors]
        
    groundTruth = pd.DataFrame({"id": list(range(len(vectors))), "file": fileNames, "author": authors})        
    vectorDataFrame = pd.DataFrame(allVectors)
    
    vectorDataFrame.to_csv(dataOut)
    groundTruth.to_csv(truthOut)
    
    with open(f'sparse.out', 'w') as fp:
        fp.writelines(sparseOutPut)
    
def parse():
    parser = argparse.ArgumentParser(description="C45")
    parser.add_argument(
        "directory", type=str, help="directory of 50-50 dataset"
    )
    parser.add_argument(
        "--out", type=str, nargs='?', default='./data.out'
    )
    parser.add_argument(
        "--truth", type=str, nargs='?', default='./truth.out'
    )


    args = vars(parser.parse_args())
    return args


def main():
    args = parse()
    dataDirectory = args["directory"]
    dataOutputLoc = args['out']
    truthOutputLoc = args['truth']
    vectorize(dataDirectory, dataOutputLoc, truthOutputLoc)
    
if __name__ == "__main__": 
    main()