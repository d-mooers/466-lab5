#!/bin/bash

echo "Bash version ${BASH_VERSION}..."

for n in 180 220 280 300
	do
	for m in 150 120 100 75
		do
		for N in 100 1000
			do
			echo "Running RF with  $n trees, $m attributes per tree, $N data points per tree"
			python RFAuthorship.py data.out --t 0 --m $m --N $n --NumDataPoints $N
		done
	done
done
