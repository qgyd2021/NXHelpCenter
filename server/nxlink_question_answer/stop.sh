#!/usr/bin/env bash

kill -9 `ps -aef | grep 'run_nxlink_question_answer.py' | grep -v grep | awk '{print $2}'`

