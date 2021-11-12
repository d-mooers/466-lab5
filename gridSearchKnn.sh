#!/bin/bash

echo "Bash version ${BASH_VERSION}..."

for i in {20..80..2}
	do
		echo "Running KNN with k = $i"
		python knnAuthorship.py sparse-data.out --k $i
	done
