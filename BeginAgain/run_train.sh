export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver/
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

if [ -d ./output ];then
    rm -rf ./output
    mkdir -p ./output
else
    mkdir -p ./output
fi

echo "start finetune..."
torchrun --nproc_per_node=8 --master_port=36751 /work/home/BeginAgain13/train.py \
    --model_name_or_path /work/mount/presetdata/openmind/qwen1.5_7b \
    --data_path /work/home/BeginAgain13/dataset.json \
    --deepspeed /work/home/BeginAgain13/ds_config.json \
    --bf16 True \
    --output_dir /work/home/BeginAgain13/output/Qwen1_5 \
    --overwrite_output_dir \
    --max_steps 200 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --model_max_length 512 \
    --seed 1234 \
    --logging_steps 1 > /work/home/BeginAgain13/output/finetune_qwen1.5_7b.log 2>&1 &
wait 