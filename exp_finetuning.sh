# PYTHONPATH=/workspace/smallcap_concat CUDA_VISIBLE_DEVICES=1 python train.py --i2t_features_dir /workspace/smallcap_concat/l14_features --features_dir /workspace/smallcap_concat/features --mlp_path /workspace/smallcap_concat/pic2word.pt --experiments_dir cat_openai_finetuning --train_mlp
# PYTHONPATH=/workspace/smallcap_concat CUDA_VISIBLE_DEVICES=1 python train.py --i2t_features_dir /workspace/smallcap_concat/l14_features --features_dir /workspace/smallcap_concat/features --mlp_path /workspace/smallcap_concat/pic2word.pt --experiments_dir cat_openai_freeze

i2t_features_dir="$2"
experiment_name="${1}_${i2t_features_dir}_finetune"

echo "${experiment_name}"

PYTHONPATH=/data/public/polar/ws/smallcap_concat python train.py \
  --i2t_features_dir /data/public/polar/ws/smallcap_concat/${i2t_features_dir} \
  --features_dir /data/public/polar/ws/smallcap_concat/features \
  --mlp_path /data/public/polar/ws/smallcap_concat/pic2word.pt \
  --experiments_dir ${experiment_name} \
  --train_mlp \
  --num_workers 32 --lr 1e-3 --batch_size 256

for var in 2214 4428 6642 8856 11070 13284 15498 17712 19926 22140
do
  echo "val : ${experiment_name}_${var}"
  PYTHONPATH=/data/public/polar/ws/smallcap_concat python infer.py \
                  --i2t_features_path /data/public/polar/ws/smallcap_concat/${i2t_features_dir}/val.hdf5 \
                  --features_path /data/public/polar/ws/smallcap_concat/features/val.hdf5 \
                  --model_path /data/public/polar/ws/smallcap_concat/${experiment_name}/rag_7M_gpt2 \
                  --checkpoint_path checkpoint-${var}
  python coco-caption/run_eval.py coco-caption/annotations/captions_valKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-${var}/val_preds.json" > "${experiment_name}/val_${var}.txt"
done

for var in 2214 4428 6642 8856 11070 13284 15498 17712 19926 22140
do
  echo "test : ${experiment_name}_${var}"
  PYTHONPATH=/data/public/polar/ws/smallcap_concat python infer.py \
                  --i2t_features_path /data/public/polar/ws/smallcap_concat/${i2t_features_dir}/test.hdf5 \
                  --features_path /data/public/polar/ws/smallcap_concat/features/test.hdf5 \
                  --model_path /data/public/polar/ws/smallcap_concat/${experiment_name}/rag_7M_gpt2 \
                  --checkpoint_path checkpoint-${var} \
                  --infer_test
  python coco-caption/run_eval.py coco-caption/annotations/captions_testKarpathy.json "${experiment_name}/rag_7M_gpt2/checkpoint-${var}/test_preds.json" > "${experiment_name}/test_${var}.txt"
done