TRAINER="HSP-SAM"

DATASET=("CVC-ClinicDB" "UDIAT" "dsb-2018" "isic2018")
ROOT_PATH="/root/autodl-tmp/data"
OUTPUT_PATH="/root/autodl-tmp/save/${TRAINER}"
CONFIG="lr_0.0012_weight_decay_0.1"

for dataset in ${DATASET[@]}
do
    DATA_PATH="${ROOT_PATH}/${dataset}"
    OUTPUT="${OUTPUT_PATH}/${dataset}/${CONFIG}"
    VISUAL_PATH="/root/autodl-tmp/visualization/${TRAINER}/${dataset}"

    python train.py \
        --data_path=${DATA_PATH} \
        --output=${OUTPUT} \
        --visual_path=${VISUAL_PATH} \
        --root_path=${DATA_PATH} \
        --num_classes=1 \
        --batch_size=20 \
        --n_gpu=2 \
        --base_lr=0.0012 \
        --weight_decay=0.1 \
        --vit_name='vit_h' \
        --ckpt='/root/autodl-tmp/pretrained/sam_vit_h_4b8939.pth' 
done