export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver/

if [ -d ./output ];then
    rm -rf ./output
    mkdir -p ./output
else
    mkdir -p ./output
fi

echo "start finetune..."
torchrun --nproc_per_node=8 --master_port=36751 /work/home/NEW_GAME/new_game.py \
    --model_name_or_path /work/mount/presetdata/openmind/qwen1.5_7b \
    --data_path /work/home/NEW_GAME/mixed_flavor.json \
    --eval_data_path /work/home/NEW_GAME/final_data.json \
    --deepspeed /work/home/NEW_GAME/ds_config.json \
    --bf16 True \
    --output_dir /work/home/NEW_GAME/output/Qwen1_5 \
    --overwrite_output_dir \
    --max_steps 200 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --model_max_length 512 \
    --seed 1234 \
    --logging_steps 1 > /work/home/NEW_GAME/output/finetune_qwen1.5_7b.log 2>&1 &
wait