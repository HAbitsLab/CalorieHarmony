#!/bin/bash

python  preprocessing.py "/output_files/sampleData" "P401 P404 P405 P409 P410 P411 P415 P417 P418 P419 P421 P422 P423 P424 P425 P427 P429 P431" "In Lab" &&
python preprocessing.py "/output_files/sampleData" "P404 P405 P409 P410 P411 P412 P415 P416 P417 P418 P419 P420 P421 P423 P424 P425 P427 P429 P431" "In Wild" &&
python cv.py "/output_files/sampleData" "P401 P404 P405 P409 P410 P411 P415 P417 P418 P421 P422 P423 P424 P425 P427 P429 P431" &&
python compare_wild.py "/output_files/sampleData" "P404 P405 P409 P410 P411 P412 P415 P416 P417 P418 P419 P420 P421 P423 P424 P425 P427 P429 P431" &&
python stat_for_paper.py > "/output_files/stat_for_paper.txt"