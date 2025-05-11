#!/bin/bash

sigmas=(0.00001 0.0001 0.001 0.003 0.005 0.007 0.01 0.015 0.02 0.03 0.1 0.3 1)

for sigma in "${sigmas[@]}"
do
    echo "=========================================="
    echo "running full pipeline for sigma = $sigma"
    echo "=========================================="

    SECONDS=0

    echo "model training: knn"
    python train_KNN.py $sigma

    

    duration=$SECONDS
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    echo "pipeline execution completed for sigma = $sigma"
    echo "elapsed time: ${minutes}m ${seconds}s"
    echo ""
done

echo "all sigma values processed successfully."
