python main.py \
  --savedir './results/maml/mimgnet_5shot' \
  --dataset 'mimgnet' \
  --mode 'meta_train' \
  --gpu_id 0 \
  --metabatch 4 \
  --inner_lr 0.01 \
  --n_steps 5 \
  --way 5 \
  --shot 5 \
  --query 15 \
  --meta_lr 1e-4 \
  --n_test_mc_samp 1 \
  --maml

python main.py \
  --savedir './results/maml/mimgnet_5shot' \
  --dataset 'mimgnet' \
  --mode 'meta_test' \
  --gpu_id 0 \
  --metabatch 5 \
  --inner_lr 0.01 \
  --n_steps 5 \
  --way 5 \
  --shot 5 \
  --query 15 \
  --meta_lr 1e-4 \
  --n_test_mc_samp 1 \
  --maml
