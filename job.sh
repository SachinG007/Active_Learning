#!/bin/sh
## run command 10 times 
for i in 1 2 3 4 5

do
	iterNum="$i"

cmd="python CIFAR_CoreSet.py --iterNum $iterNum "
#echo $cmd
$cmd
done
