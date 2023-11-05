
# PYTHONPATH=/workspace/smallcap_concat CUDA_VISIBLE_DEVICES=1 python train.py --i2t_features_dir /workspace/smallcap_concat/l14_features --features_dir /workspace/smallcap_concat/features --mlp_path /workspace/smallcap_concat/pic2word.pt --experiments_dir cat_openai_finetuning --train_mlp
# PYTHONPATH=/workspace/smallcap_concat CUDA_VISIBLE_DEVICES=1 python train.py --i2t_features_dir /workspace/smallcap_concat/l14_features --features_dir /workspace/smallcap_concat/features --mlp_path /workspace/smallcap_concat/pic2word.pt --experiments_dir cat_openai_freeze

experiment_name="$1"
gpu="$2"

for var in 8856 17712 26568 35424 44280 53136 61992 70848 79704 88560
do
  PYTHONPATH=/workspace/smallcap_concat CUDA_VISIBLE_DEVICES=${gpu} python infer.py \
                  --i2t_features_path /workspace/smallcap_concat/l14_features/val.hdf5 \
                  --features_path /workspace/smallcap_concat/features/val.hdf5 \
                  --model_path /workspace/smallcap_concat/${experiment_name}/rag_7M_gpt2 \
                  --checkpoint_path checkpoint-${var}
  CUDA_VISIBLE_DEVICES=${gpu} python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}/val_${var}.txt"

  # CUDA_VISIBLE_DEVICES=0 python infer.py --encoder_name "openai/clip-vit-large-patch14" --model_path "${experiment_name}_freeze/rag_7M_gpt2" --checkpoint_path checkpoint-${var} --infer_test
  # CUDA_VISIBLE_DEVICES=0 python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}_freeze/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}_freeze/results/test_${var}.txt"
done
