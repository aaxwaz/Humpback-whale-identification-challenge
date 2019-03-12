# Humpback whale re-identification using Siamese neural nets

Code for 5th place winning solution

## Requirements

* Hardware: GPU NVIDIA 1080 Ti
* Software: Python 3.6, keras==2.2.4, keras-retinanet==0.5.0, albumentations, pyvips, scipy, numpy, pandas, tqdm, lap, sklearn

## Input data location

1) Both training and test images should be put inside below folder separately: 
* `../data/train/`
* `../data/test/`

2) train.csv and sample_submission.csv are at below locations: 
* `../data/train.csv`
* `../data/sample_submission.csv`

## Part 1 - Bounding box models 

* Requires: `../modified_data/p2bb_v5.pkl`
* Requires: `../modified_data/retinanet/cropping_train_v2.csv` - some boxes for playground competition

1) `python3 retinanet/r10_create_csv_for_retinanet.py`
2) `python3 retinanet/r30_train_backbone_resnet152_kfold.py`
3) `python3 retinanet/r31_convert_retinanet_model.py`
4) `python3 retinanet/r31_get_vectors_backbone_resnet152_kfold.py`
5) `python3 retinanet/r32_average_boxes.py`

As result we obtain following files:
* `../modified_data/p2bb_averaged_v1.pkl` - boxes for train/test images
* `../modified_data/p2bb_averaged_playground_v1.pkl` - boxes for playground images

## Part 2 - Siamese Nets with DenseNet121 and SE-ResNext50

### Generate KFold splits
6) `python3 r10_create_kfold_split.py`

As result we have 2 files with different KFold splits
* `../modified_data/kfold/new_4_folds_split_train_val_v1.pkl` - kfold split v1 (used by DenseNet121)
* `../modified_data/kfold/new_4_folds_split_train_val_v2.pkl` - kfold split v2 (used by SE-ResNext50)

### Part with siamese nets (DenseNet121)
7) `python3 siamese_net_v5_densenet121/r10_seamese_net_warmstart_from_scratch_224px.py`
8) `python3 siamese_net_v5_densenet121/r11_seamese_net_warmstart_finetune_384px.py`
9) `python3 siamese_net_v5_densenet121/r15_seamese_net_train_v5_finetune_384px.py 0`
10) `python3 siamese_net_v5_densenet121/r15_seamese_net_train_v5_finetune_384px.py 1`
11) `python3 siamese_net_v5_densenet121/r15_seamese_net_train_v5_finetune_384px.py 2`
12) `python3 siamese_net_v5_densenet121/r15_seamese_net_train_v5_finetune_384px.py 3`
13) `python3 siamese_net_v5_densenet121/r15_seamese_net_train_v5_finetune_512px.py 0`
14) `python3 siamese_net_v5_densenet121/r15_seamese_net_train_v5_finetune_512px.py 1`
15) `python3 siamese_net_v5_densenet121/r15_seamese_net_train_v5_finetune_512px.py 2`
16) `python3 siamese_net_v5_densenet121/r15_seamese_net_train_v5_finetune_512px.py 3`
17) `python3 siamese_net_v5_densenet121/r26_seamese_net_inference_v5_512px.py`

### Part with siamese nets (SE-ResNext50)
18) `python3 siamese_net_v6_se_resnext/r11_seamese_net_warmstart_from_scratch_224px.py`
19) `python3 siamese_net_v6_se_resnext/r12_seamese_net_warmstart_from_scratch_384px.py`
20) `python3 siamese_net_v6_se_resnext/r15_seamese_net_train_v6_finetune_384px.py 0`
21) `python3 siamese_net_v6_se_resnext/r15_seamese_net_train_v6_finetune_384px.py 1`
22) `python3 siamese_net_v6_se_resnext/r15_seamese_net_train_v6_finetune_384px.py 2`
23) `python3 siamese_net_v6_se_resnext/r15_seamese_net_train_v6_finetune_384px.py 3`
24) `python3 siamese_net_v6_se_resnext/r16_seamese_net_inference_v6_384px.py`

### Create tables for using models predictions in ensemble
25) `python3 r20_prepare_matrices_for_ensemble.py`

As result we will have 4 files with prediction matrices, which will be used for ensemble
* `../features/cv-analysis-fs14-LB959-densenet121-512px-sparse.pkl`
* `../features/cv-analysis-fs14-LB959-densenet121-512px-sparse-test.pkl`
* `../features/cv-analysis-fs16-LB959-seresnext50-384px-sparse.pkl`
* `../features/cv-analysis-fs16-LB959-seresnext50-384px-sparse-test.pkl`

## Part 3 - Kernel Siamese Nets training pipeline

### Create kfold splits 
* `python kfold_splits_for_kernel.py`

### Kerenl siamese net training

Train four fold siamese training, each training requiring two GPUs. Make sure you have enough GPUs (8) to run all four model training parallelly. Otherwise, run in sequence four times

* `python snn_train_kernel_384_to_1024.py --CUDA_VISIBLE_DEVICES 0,1 --RUN_FOLD 0`
* `python snn_train_kernel_384_to_1024.py --CUDA_VISIBLE_DEVICES 2,3 --RUN_FOLD 1`
* `python snn_train_kernel_384_to_1024.py --CUDA_VISIBLE_DEVICES 4,5 --RUN_FOLD 2`
* `python snn_train_kernel_384_to_1024.py --CUDA_VISIBLE_DEVICES 6,7 --RUN_FOLD 3`

### Kernel net inference

After above trainings are done, find out the best saved weights from each model based on log file, and run inference below to generate the final averaged test-vs-train score matrix 

* `python snn_inference_kernel_1024.py --model_weights_1 ../path_to_your_best_weights_1 --model_weights_2 ../path_to_your_best_weights_2 --model_weights_3 ../path_to_your_best_weights_3 --model_weights_4 ../path_to_your_best_weights_4`

## Part 4 - Ensemble

Final ensemble of all models with post processing steps to generate final submit

1) Check to make sure all three models are generated inside ../features/, then run: 
* `python final_ensemble_with_post_proc.py`

2) Final submit will be generated in: 
* `../submission/final_submit_with_post_proc.csv`
