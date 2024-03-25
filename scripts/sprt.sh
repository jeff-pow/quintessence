#!/bin/bash

# rm stderr.txt
# rm pgnout.txt
# rm nohup.out

/home/jeff/ob-client/cutechess-ob \
-engine name=dev cmd=./Titan \
-engine name=main cmd=./Titan \
-games 2 -rounds 250 \
-pgnout "pgnout.txt" \
-sprt elo0=0.0 elo1=3.0 alpha=0.05 beta=0.05 \
-each proto=uci tc=8+0.08 stderr=stderr.txt \
-openings order=random file="/home/jeff/ob-client/Books/Pohl.epd" format=epd \
-concurrency 6 \
-ratinginterval 10 \
# -debug \
