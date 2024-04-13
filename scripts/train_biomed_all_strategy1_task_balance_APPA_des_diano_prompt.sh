start_time=$(date +%s)

folder="./log/"

# 检测文件夹是否存在
if [ ! -d "$folder" ]; then
    echo "${folder} dose not exist, build it..."
    # 如果文件夹不存在，则创建文件夹
    mkdir -p "$folder"
    echo "Folder created successfully"
else
    echo "./log/ folder exist"
fi

time=$(date "+%Y_%m_%d_%H-%M")
echo -e "train model with configuration:\n1. Backbone: biomedclip\n2. Loss: contrastive, orthoginal, graph loss\n\
3. Labeling strategy: S1 stragety--binary classification.\n4. Prompt: descript + diagnotic prompt\n5. Weghting stragety: task balance\n\
6. log: <....>/output/log/biomed_all_task_balance_S1_des_diago_prompt_${time}.log\n\
7. using AP—PA—data"

python ../src/main.py \
-b biomedclip \
-ho binary \
-ws task_balance  \
-LS S1 \
--prompt dis_diag_des \
--AP-PA-view >> "./log/biomed_all_task_balance_S1_des_diago_prompt_${time}.log"

end_time=$(date +%s)

duration=$((end_time - start_time))

hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "脚本运行时间为：$hours 小时 $minutes 分钟 $seconds 秒"
