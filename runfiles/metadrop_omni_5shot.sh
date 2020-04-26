python main.py \
  --savedir './results/metadrop/omni_5shot' \
  --dataset 'omniglot' \
  --mode 'meta_train' \
  --gpu_id 4 \
  --metabatch 4 \
  --n_steps 5 \
  --inner_lr 0.4 \
  --way 20 \
  --shot 5 \
  --query 15 \
  --n_train_iters 60000 \
  --meta_lr 1e-3 \
  --n_test_mc_samp 1

python main.py \
  --savedir './results/metadrop/omni_5shot' \
  --dataset 'omniglot' \
  --mode 'meta_test' \
  --gpu_id 4 \
  --metabatch 1 \
  --n_steps 5 \
  --inner_lr 0.4 \
  --way 20 \
  --shot 5 \
  --query 15 \
  --n_train_iters 60000 \
  --meta_lr 1e-3 \
  --n_test_mc_samp 30
