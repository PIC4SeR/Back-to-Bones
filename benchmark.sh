#!/bin/bash
    
PACS="PACS"
VLCS="VLCS"
OfficeHome="OfficeHome"
TerraIncognita="TerraIncognita"
train="False"
    
for model in "vit_base16" "deit_base16" "convit_base"; do # "vit_base16" "deit_base16" "convit_base"...

    for meth in "None" "AGGD" "Mixup" "CORAL" "RSC"; do # None RSC Mixup CORAL...

        for dataset in "PACS" "VLCS" "OfficeHome" "TerraIncognita"; do # "PACS" "VLCS" "OfficeHome" "TerraIncognita"
            
            if [ "$dataset" = "$PACS" ]; then
                domains="photo art_painting cartoon sketch" # "photo art_painting cartoon sketch"
                lr=0.00001
            elif [ "$dataset" = "$VLCS" ]; then
                domains="CALTECH LABELME PASCAL SUN" # "CALTECH LABELME PASCAL SUN"
                lr=0.00001
            elif [ "$dataset" = "$OfficeHome" ]; then
                domains="Product Art Clipart Real_World" # "Product Art Clipart Real_World"
                lr=0.00001
            elif [ "$dataset" = "$TerraIncognita" ]; then
                domains="100 38 43 46"  # "100 38 43 46" 
                lr=0.000008
            fi
            
            for target in $domains; do # Iterate on domains
        
                for i in 1 2 3; do # Multiple runs
                
                    date
                    echo "Training: model=$model, meth=$meth, dataset=$dataset, target=$target"
                
                    python3 train.py --lr $lr --lr_sched cos --network $model --batch_size 32 --epochs 30 --cuda 0 --dataset $dataset --target $target --optimizer Adam --meth $meth --verbose True >> results/${model}_${dataset}_${target}_${meth}.txt
                    
                done
            done
        done
    done
done