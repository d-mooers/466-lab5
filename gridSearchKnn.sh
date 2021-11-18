#!/bin/bash

echo "Bash version ${BASH_VERSION}..."

for i in {1..20..5}
	do
		echo "Running KNN with k = $i"
		python knnAuthorship.py sparse-data.out --k $i --metric cosine
	done
