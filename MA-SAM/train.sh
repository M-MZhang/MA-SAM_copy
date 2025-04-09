TRAINER="HSP-SAM"

DATASET=("DRIVE" "CVC-ClinicDB" "UDIAT" "dsb-2018" "isic2018")
ROOT_PATH="/root/data1/zmm/seg4medicine/data"
OUTPUT_PATH="/root/data1/zmm/seg4medicine/save/${TRAINER}"
CONFIG="lr_0.0012"

for dataset in ${DATASET[@]}
do
    DATA_PATH="${ROOT_PATH}/${dataset}"
    OUTPUT="${OUTPUT_PATH}/${dataset}/${CONFIG}"
    VISUAL_PATH="/root/data1/zmm/seg4medicine/visualization/${dataset}"

    CUDA_VISIBLE_DEVICES=0,1 python3 train.py \
        --data_path=${DATA_PATH} \
        --output=${OUTPUT} \
        --visual_path=${VISUAL_PATH} \
        --root_path=${DATA_PATH} \
        --num_classes=1 \ 
        --batch_size=8 \
        --n_gpu=2 \
        --base_lr=0.0012 
done