#!/bin/bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ${SCRIPT_DIR}/dev-v1.1.json
wget https://zenodo.org/record/3750364/files/vocab.txt?download=1 -O ${SCRIPT_DIR}/vocab.txt
