TRAINER="HSP-SAM"

Source_DATASET=("DRIVE" "isic2018" "CVC-ClinicDB" "UDIAT" "dsb-2018" )
Target_DATASET=("STARE" "PH2" "CVC-ColonDB" "BUSI" "TNBC" )
ROOT_PATH="/root/autodl-tmp/data"
OUTPUT_PATH="/root/autodl-tmp/save/${TRAINER}"
CONFIG="lr_0.0008_weight_decay_0.1"
prompt_num=(1 2 4 8 16)


for ((i=0;i<${#Source_DATASET[@]};i++))
do
    for prompt in ${prompt_num[@]}
    do
    DATA_PATH="${ROOT_PATH}/${Target_DATASET[i]}"
    adapt_ckpt="${OUTPUT_PATH}/${Source_DATASET[i]}/${CONFIG}/best.pth"
    OUTPUT="${OUTPUT_PATH}/${Target_DATASET[i]}/${CONFIG}"/${prompt}
    VISUAL_PATH="/root/autodl-tmp/visualization/${TRAINER}/${Target_DATASET[i]}"

    CUDA_VISIBLE_DEVICES=0,1 python test.py \
        --data_path=${DATA_PATH} \
        --output_dir=${OUTPUT} \
        --visual_path=${VISUAL_PATH} \
        --n_gpu=2 \
        --batch_size=20 \
        --num_classes=1 \
        --num_prompts=${prompt} \
        --vit_name='vit_h' \
        --ckpt='/root/autodl-tmp/pretrained/sam_vit_h_4b8939.pth'\
        --adapt_ckpt=${adapt_ckpt} 
    done
done