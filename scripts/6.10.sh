echo "training vision only model with biomedclip model -- binary cross-entropy loss"
python ../src/main.py --backbone biomedclip  --trainable_VisionEncoder  --AP-PA-view  --labeling_strategy S1 --backbone biomedclip  -vo >> ../src/6_10/6.10_vo_binary_cross_entropy.log

echo "training vision only model with biomedclip model -- focal loss"
python ../src/main.py --backbone biomedclip  --trainable_VisionEncoder  --AP-PA-view  --labeling_strategy S1 --backbone biomedclip  --focal_loss -vo >> ../src/6_10/6.10_vo_focal_loss.log

echo "training with aligment -- binary cross-entropy loss"
python ../src/main.py --backbone biomedclip  --trainable_VisionEncoder  --AP-PA-view  --labeling_strategy S1 --backbone biomedclip  --focal_loss --high_order binary >> ../src/6_10/6.10_aligment_binary_cross_entropy.log

echo "training with aligment -- focal loss"
python ../src/main.py --backbone biomedclip  --trainable_VisionEncoder  --AP-PA-view  --labeling_strategy S1 --backbone biomedclip  --focal_loss --high_order binary >> ../src/6_10/6.10_aligment_focal_loss.log





