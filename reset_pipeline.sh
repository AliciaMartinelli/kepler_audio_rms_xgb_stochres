#!/bin/bash

echo "this will delete all generated data, features, results and plots."
read -p "are you sure you want to continue? [y/N] " confirm

if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "aborted."
    exit 1
fi

rm -rf aug_dataset/
rm -rf rms/
rm -rf results/

echo "all generated folders deleted successfully."
