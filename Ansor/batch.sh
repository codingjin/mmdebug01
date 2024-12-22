#!/bin/bash

python mm_ansor.py 0 4096 4096 4096 2>&1 | tee 0_4096outlog

python mm_ansor.py 8 4096 4096 4096 2>&1 | tee 8_4096outlog

python mm_ansor.py 12 4096 4096 4096 2>&1 | tee 12_4096outlog

python mm_ansor.py 16 4096 4096 4096 2>&1 | tee 16_4096outlog

python mm_ansor.py 20 4096 4096 4096 2>&1 | tee 20_4096outlog


