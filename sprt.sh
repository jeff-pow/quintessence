#!/bin/bash

rm stderr.txt
rm pgnout.txt
rm nohup.out

nohup cutechess-cli \
-engine name=dev cmd=./Quintessence \
-engine name=main cmd=./main \
-games 2 -rounds 50000 \
-pgnout "pgnout.txt" \
-sprt elo0=-2.0 elo1=1.0 alpha=0.05 beta=0.05 \
-each proto=uci tc=8+0.08 stderr=stderr.txt \
-openings order=random file="Pohl.pgn" format=pgn \
-concurrency 6 \
-ratinginterval 10 \
# -debug \
