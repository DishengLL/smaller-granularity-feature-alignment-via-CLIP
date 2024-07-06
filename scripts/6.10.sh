echo "training vision only model with biomedclip model -- binary cross-entropy loss"
python ../src/main.py --backbone biomedclip  --trainable_VisionEncoder  --AP-PA-view  --labeling_strategy S1 --backbone biomedclip  -vo >> ../src/6_10/6.10_vo_binary_cross_entropy.log

echo "training vision only model with biomedclip model -- focal loss"
python ../src/main.py --backbone biomedclip  --trainable_VisionEncoder  --AP-PA-view  --labeling_strategy S1 --backbone biomedclip  -vo --focal_loss >> ../src/6_10/6.10_vo_focal_loss.log



# echo "training with aligment -- focal loss"
# python ../src/main.py --backbone biomedclip  --trainable_VisionEncoder  --AP-PA-view  --labeling_strategy S1 --backbone biomedclip  --focal_loss --high_order binary >> ../src/6_10/6.10_aligment_focal_loss.log


# echo "training with aligment -- binary cross-entropy loss"
# python ../src/main.py --backbone biomedclip  --trainable_VisionEncoder  --AP-PA-view  --labeling_strategy S1 --backbone biomedclip  --high_order binary >> ../src/6_10/6.10_aligment_binary_cross_entropy.log


# echo "training with aligment -- binary cross-entropy loss + task balanced"
# python ../src/main.py --backbone biomedclip  --trainable_VisionEncoder  --AP-PA-view  --labeling_strategy S1 --backbone biomedclip  --high_order binary --weight_strategy task_balance >> ../src/6_10/6.10_aligment_binary_cross_entropy_taskBalanced.log


# echo "training with aligment -- binary cross-entropy loss + uncertainty"
# python ../src/main.py --backbone biomedclip  --trainable_VisionEncoder  --AP-PA-view  --labeling_strategy S1 --backbone biomedclip  --high_order binary --weight_strategy uncertain_based_weight >> ../src/6_11/6.11_aligment_binary_cross_entropy_uncertainty.log


# echo "training with aligment -- focal loss + uncertainty"
# python ../src/main.py --backbone biomedclip  --trainable_VisionEncoder  --AP-PA-view  --labeling_strategy S1 --backbone biomedclip  --high_order binary --weight_strategy uncertain_based_weight  --focal_loss >> ../src/6_11/6.11_aligment_focal_loss_uncertainty.log