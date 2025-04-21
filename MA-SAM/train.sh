TRAINER="HSP-SAM"

DATASET=("DRIVE" "CVC-ClinicDB" "UDIAT" "dsb-2018" )
prompt_num=(1 2 4 8 16)
ROOT_PATH="/root/autodl-tmp/data"
OUTPUT_PATH="/root/autodl-tmp/save/${TRAINER}"
CONFIG="lr_0.0008_weight_decay_0.1"

for dataset in ${DATASET[@]}
do
    for prompt in ${prompt_num[@]}
    do
        DATA_PATH="${ROOT_PATH}/${dataset}"
        OUTPUT="${OUTPUT_PATH}/${dataset}/${CONFIG}/${prompt}"
        VISUAL_PATH="/root/autodl-tmp/visualization/${TRAINER}/${dataset}"

        CUDA_VISIBLE_DEVICES=0,1 python train.py \
            --num_classes=${prompt} \
            --data_path=${DATA_PATH} \
            --output=${OUTPUT} \
            --visual_path=${VISUAL_PATH} \
            --root_path=${DATA_PATH} \
            --num_classes=${prompt} \
            --batch_size=20 \
            --n_gpu=2 \
            --base_lr=0.0008 \
            --weight_decay=0.1 \
            --vit_name='vit_h' \
            --ckpt='/root/autodl-tmp/pretrained/sam_vit_h_4b8939.pth' 
    done
done