now=$(date +"%Y%m%d_%H%M%S")
root_path=/media/llog/AGMM-SASSdata/predict/sparse_budget2/full
save_path=$root_path/DINOV2b_84.36_full
resume_model=/media/llog/AGMM-SASSdata/model_files/UTCsa/full0902_dinov2b/DINOV2b_84.36.pth
mkdir -p $save_path

python eval.py \
    --resume_model=$resume_model \
    --save-mask-path=$save_path | tee $save_path/$now.txt

python eval/AMP_eval.py \
    --folder_path=$save_path | tee -a $save_path/$now.txt