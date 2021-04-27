#!/bin/bash
for ((i = 0; i < 10; i+=1));
do
    sbatch --export=ALL,A=`expr $i` "/N/u/bdahlber/Carbonate/Desktop/visualbraking.script"
done
