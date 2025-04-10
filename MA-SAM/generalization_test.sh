TRAINER="HSP-SAM"

Source_DATASET=("isic2018" "DRIVE" "CVC-ClinicDB" "UDIAT" "dsb-2018" )
Target_DATASET=("PH2" "STARE" "CVC-ColonDB" "BUSI" "TNBC" )
ROOT_PATH="/root/autodl-tmp/data"
OUTPUT_PATH="/root/autodl-tmp/save/${TRAINER}"


for ((i=0;i<${#Source_DATASET[@]};i++))
do
    DATA_PATH="${ROOT_PATH}/${Target_DATASET[i]}"
    lora_ckpt="${OUTPUT_PATH}/${Source_DATASET[i]}/best.pth"
    OUTPUT="${OUTPUT_PATH}/${Target_DATASET[i]}"
    VISUAL_PATH="/root/autodl-tmp/visualization/${TRAINER}/${Target_DATASET[i]}"

    python test.py \
        --data_path=${DATA_PATH} \
        --output_dir=${OUTPUT} \
        --visual_path=${VISUAL_PATH} \
        --n_gpu=2 \
        --batch_size=24 \
        --num_classes=1 \
        --vit_name='vit_h' \
        --ckpt='/root/autodl-tmp/pretrained/sam_vit_h_4b8939.pth'\
        --lora_ckpt=${lora_ckpt} 
done