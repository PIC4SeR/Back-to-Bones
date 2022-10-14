#!/bin/bash
    
PACS="PACS"
VLCS="VLCS"
OfficeHome="OfficeHome"
TerraIncognita="TerraIncognita"
ColoredMNIST="ColoredMNIST"
    
for model in "mnist_cnn"; do 

    for dataset in "ColoredMNIST"; do # "PACS" "VLCS" "OfficeHome" "TerraIncognita"

        if [ "$dataset" = "$PACS" ]; then
            domains="art_painting cartoon sketch" # "photo art_painting cartoon sketch"
            lr=0.00001
        elif [ "$dataset" = "$VLCS" ]; then
            domains="CALTECH LABELME PASCAL SUN" # "CALTECH LABELME PASCAL SUN"
            lr=0.00001
        elif [ "$dataset" = "$OfficeHome" ]; then
            domains="Product Art Clipart Real_world" # "Product Art Clipart Real_world"
            lr=0.00001
        elif [ "$dataset" = "$TerraIncognita" ]; then
            domains="100 38 43 46" # "100 38 43 46" 
            lr=0.000008
        elif [ "$dataset" = "$ColoredMNIST" ]; then
            domains="test"
            lr=0.001
        fi

        for target_domain in $domains; do # Iterate on domains

            for i in 0 1 2; do # Multiple runs

                date
                echo "Training: model=$model, dataset=$dataset, target=$target_domain, trial=$i"

                python3 train.py --learning_rate $lr --lr_sched const --network $model --batch_size 32 --epochs 30 --cuda 0 --dataset $dataset --target_domain $target_domain --optimizer SGD --trial $i --image_size 28 >> results/${model}_${dataset}.txt
                    
            done
        done
    done
done
