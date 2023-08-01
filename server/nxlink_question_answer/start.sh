#!/usr/bin/env bash

rm -rf nohup.out
rm -rf logs/

source /data/local/bin/NXHelpCenter/bin/activate

nohup python3 run_nxlink_question_answer.py > nohup.out &
