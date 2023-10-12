# python maskcut.py --vit-arch base --patch-size 8 --tau 0.15 --fixed_size 480 --N 3 --num-folder-per-job 1000 --job-index 0 --dataset-path

export DETECTRON2_DATASETS=/home/kate.brillantes/thesis/cutler/datasets
python train_net.py --num-gpus 1 --config-file model_zoo/configs/CutLER-Selected/cascade_mask_rcnn_R_50_FPN.yaml
python combine_json.py --basedir /home/kate.brillantes/thesis/cutler/cutler/output_cvat/inference

python tools/get_self_training_ann.py --new-pred output_cvat/inference/coco_instances_results.json --prev-ann /home/kate.brillantes/thesis/cutler/datasets/selected_cvat2023/annotations/selected_train.json --save-path /home/kate.brillantes/thesis/cutler/datasets/selected_cvat2023/annotations/cutler_selectedcvat_train_r1.json  --threshold 0.7
python train_net.py --num-gpus 1 --config-file model_zoo/configs/CutLER-Selected/cascade_mask_rcnn_R_50_FPN_self_train_r1.yaml MODEL.WEIGHTS output/model_final.pth OUTPUT_DIR output/self-train-r1/ # path to save checkpoints
python combine_json.py --basedir /home/kate.brillantes/thesis/cutler/cutler/output_cvat/self-train-r1/inference


python tools/get_self_training_ann.py --new-pred output_cvat/self-train-r1/inference/coco_instances_results.json --prev-ann /home/kate.brillantes/thesis/cutler/datasets/selected_cvat2023/annotations/selected_train.json --save-path /home/kate.brillantes/thesis/cutler/datasets/selected_cvat2023/annotations/cutler_selectedcvat_train_r2.json  --threshold 0.65
python train_net.py --num-gpus 1 --config-file model_zoo/configs/CutLER-Selected/cascade_mask_rcnn_R_50_FPN_self_train_r1.yaml MODEL.WEIGHTS output/model_final.pth OUTPUT_DIR output/self-train-r1/ # path to save checkpoints
