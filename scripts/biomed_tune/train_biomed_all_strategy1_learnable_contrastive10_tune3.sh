
# since in task-balance weighting strategies, order the value of the losses from large to small
# contrastive loss > classification loss > orthogonal loss > high-order loss

# review the whole pipiline, contrastive loss is the bridge is the joint to connect textual and visual branch.
# If the path of joint is blocked, even though the orthogonal loss and high-order loss are small, the impact can not flow 
# into the image branch.

# based on this consideration, the joint between textual and visual branch should be ensured. Therefore, I try to fix the 
# weight of contrastive to emphasize this contrastive loss

# in this script, I try the weight of contrastiv loss to be 10, and which is fixed during the whole proccesure.


# previously, my experiment runs on task-balanced strategy, which using the each loss itself as the parameter of weighting
# parameter. In this setting, I want to emphasize the importance of contrastive loss. 
# therefore, I use learnable weighting strategy

# fine tuning the last 3 att blocks in VIT

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
3. Labeling strategy: S1 stragety--binary classification.\n4. Prompt: basic prompt\n5. Weghting stragety: learnable\n\
6. log: <....>/output/log/biomed_all_learnable_weight_S1_alpah10_${time}.log\n\
7. contrastive parameter (alpha) : 10\n\
8. tune last 3 attention blocks"

python ../src/main.py \
-b biomedclip \
-ho binary \
--learnable_weight \
-LS S1 \
-CP 10 \
-TP 3 \
>> "./log/biomed_all_learnable_weight_S1_alpah10_tunn3_${time}.log"

end_time=$(date +%s)

duration=$((end_time - start_time))

hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "脚本运行时间为：$hours 小时 $minutes 分钟 $seconds 秒"
