#!/bin/bash

echo "Bash version ${BASH_VERSION}..."

for file in knnAuthorship/*.out
	do
		name=${file##*/}
		python classifierEvaluation.py $file  truth.out --out matrix/$name.out > results/$name.out
	done
