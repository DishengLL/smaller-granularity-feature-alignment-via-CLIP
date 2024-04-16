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
echo -e "train model with configuration:\n1. Backbone: densenet\n2. Loss: contrastive, high-order, orthoginal, classification\n\
3. Labeling strategy: S1 stragety--binary classification.\n\
6. log: <....>/output/log/densenet_contrastive_S1_APPA_${time}.log\n\
7. using AP—PA—data"

echo

echo "python ../src/main.py \
-bv densenet \
-ho binary \
-ws task_balance  \
-LS S1 \
--prompt dis_diag_des \
-LS S1 \
--AP-PA-view >> "./log/densenet_contrastive_S1_APPA_${time}.log""

python ../src/main.py \
-bv densenet \
-ho binary \
-ws task_balance  \
-LS S1 \
--prompt dis_diag_des \
-LS S1 \
--AP-PA-view >> "./log/densenet_contrastive_S1_APPA_${time}.log"

end_time=$(date +%s)

duration=$((end_time - start_time))

hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "脚本运行时间为：$hours 小时 $minutes 分钟 $seconds 秒"
