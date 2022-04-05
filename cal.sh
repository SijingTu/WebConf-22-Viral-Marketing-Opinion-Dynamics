#!/bin/bash

for var in  Convote, Polbooks, Netscience ;do
   julia -O3 NetProcess.jl $var 
done