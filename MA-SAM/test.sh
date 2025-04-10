TRAINER="HSP-SAM"

DATASET=("DRIVE" "CVC-ClinicDB" "UDIAT" "dsb-2018" "isic2018")
# DATASET="dsb-2018"
ROOT_PATH="/root/autodl-tmp/data"
OUTPUT_PATH="/root/autodl-tmp/save/${TRAINER}"


for dataset in ${DATASET[@]}
do
    DATA_PATH="${ROOT_PATH}/${dataset}"
    OUTPUT="${OUTPUT_PATH}/${dataset}"
    adapt_ckpt="${OUTPUT}/best.pth"
    VISUAL_PATH="/root/autodl-tmp/visualization/${TRAINER}/${dataset}"

    python test.py \
        --data_path=${DATA_PATH} \
        --output_dir=${OUTPUT} \
        --visual_path=${VISUAL_PATH} \
        --n_gpu=2 \
        --batch_size=24 \
        --num_classes=1 \
        --vit_name='vit_h' \
        --ckpt='/root/autodl-tmp/pretrained/sam_vit_h_4b8939.pth'\
        --adapt_ckpt=${lora_ckpt} 

done