TRAINER="Ft-SAM"
DATASET=("DRIVE" "CVC-ClinicDB" "UDIAT" "dsb-2018" "isic2018")

ROOT_PATH="/root/autodl-tmp/data"
OUTPUT_PATH="/root/autodl-tmp/save/${TRAINER}"
CONFIG="lr_0.0008_weight_decay_0.1"


for dataset in ${DATASET[@]}
do
    DATA_PATH="${ROOT_PATH}/${dataset}"
    OUTPUT="${OUTPUT_PATH}/${dataset}/${CONFIG}"
    adapt_ckpt="${OUTPUT}/best.pth"
    VISUAL_PATH="/root/autodl-tmp/visualization/${TRAINER}/${dataset}"

    CUDA_VISIBLE_DEVICES=0,1 python test.py \
        --data_path=${DATA_PATH} \
        --output_dir=${OUTPUT} \
        --visual_path=${VISUAL_PATH} \
        --n_gpu=2 \
        --batch_size=20 \
        --num_classes=1 \
        --vit_name='vit_h' \
        --ckpt='/root/autodl-tmp/pretrained/sam_vit_h_4b8939.pth'\
        --adapt_ckpt=${adapt_ckpt} 

done