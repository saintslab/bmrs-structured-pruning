# Source code for the paper "BMRS: Bayesian Model Reduction for Structured Pruning"

Source code to reproduce the experimental results in the paper "BMRS: Bayesian Model Reduction for Structured Pruning".

The pruning layers are defined in `modules/layers` and `modules/util`. The classifiers are defined in `modules/classifiers`.
We implement the following pruning layers:

- `LogUniformPruningLayer`: The baseline multiplicative noise layer using SNR for pruning from 
Neklyudov et al. 2017
- `LogUniformL2NormPruningLayer`: Multiplicative noise layer using the L2 norm of input weights for pruning
- `LogUniformApproximateDiracBMRPruningLayer`: $BMRS_{N}$ pruning layer using an approximate dirac spike at 0
- `LogUniformCDFBMRPruningLayer`: $BMRS_{U}$ pruning layer using a reduced log uniform distribution

The code for training with continuous pruning is in `train.py`; The code to run post-training pruning and generate
compression vs. accuracy statistics is in `compression_vs_accuracy_separate.py`; And the code to generate the plots
is in `generate_plots_accuracy_vs_compression.py`. The latex tables from the paper can be generated using 
`analysis/generate_latex_tables.py` after first training a set of models to get their performance metrics.

Running the code is dependent on the following packages and uses python 3.8 or above:

```
carbontracker==1.2.3
datasets==2.14.6
matplotilb
numpy==1.26.4
pandas==1.3.5
scikit-learn
seaborn==0.12.0
tokenizers==0.14.1
torch==2.2.0
torchvision==0.16.0
tqdm==4.56.2
transformers==4.34.1
wandb==0.15.12
```

To ignore using Weights & Biases for tracking, execute `wandb disabled` before training.

## Training a model
The options for training are given as follows:

```
usage: train.py [-h] [--run_name RUN_NAME] --metrics_dir METRICS_DIR --output_dir OUTPUT_DIR [--pruning_class {LogUniformPruningLayer,LogUniformL2NormPruningLayer,LogUniformApproximateDiracBMRPruningLayer,LogUniformCDFBMRPruningLayer}]
                [--network {mlp,lenet5,resnet18,resnet50,vit}] [--dataset {mnist,cifar10,fashion_mnist,tinyimagenet}] [--batch_size BATCH_SIZE] [--n_layers N_LAYERS] [--hidden_dim HIDDEN_DIM] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--mu_p_tilde MU_P_TILDE]
                [--n_epochs N_EPOCHS] [--seed SEED] [--disable_pruning] [--debug] [--compress_epochs COMPRESS_EPOCHS] [--kl_averaging {sum,mean}] [--pretrained] [--skip_exists] [--tags TAGS [TAGS ...]] [--compression_percentage_file COMPRESSION_PERCENTAGE_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --run_name RUN_NAME   A name for this run
  --metrics_dir METRICS_DIR
                        Directory to save metrics in
  --output_dir OUTPUT_DIR
                        Directory to save models in
  --pruning_class {LogUniformPruningLayer,LogUniformL2NormPruningLayer,LogUniformApproximateDiracBMRPruningLayer,LogUniformCDFBMRPruningLayer}
                        Name of the pruning class to use
  --network {mlp,lenet5,resnet18,resnet50,vit}
                        Name of the network to use
  --dataset {mnist,cifar10,fashion_mnist,tinyimagenet}
                        Name of the network to use
  --batch_size BATCH_SIZE
                        The batch size
  --n_layers N_LAYERS   Number of layers
  --hidden_dim HIDDEN_DIM
                        Number of units in each layer
  --learning_rate LEARNING_RATE
                        The learning rate
  --weight_decay WEIGHT_DECAY
                        Amount of weight decay
  --mu_p_tilde MU_P_TILDE
                        mu to use for new prior
  --n_epochs N_EPOCHS   The number of epochs to run
  --seed SEED           Random seed
  --disable_pruning     Disables pruning
  --debug               Debug mode
  --compress_epochs COMPRESS_EPOCHS
                        Number of epochs to wait to compress
  --kl_averaging {sum,mean}
                        How to reduce KL divergence
  --pretrained          Whether or not to use a pretrained model
  --skip_exists         Exit if metrics already exist for this run
  --tags TAGS [TAGS ...]
                        Tags to pass to wandb
  --compression_percentage_file COMPRESSION_PERCENTAGE_FILE
                        A metrics file to pull the compression percentage from. Only for SNR and magnitude pruning
```

For example, to train Lenet5 on Cifar10 using $BMRS_{N}$ as done in the paper, execute the following:

```
python train.py \
    --metrics_dir metrics/cifar10_bmrsn_lenet5 \
    --output_dir output/cifar10_bmrsn_lenet5 \
    --tags cifar10 bmrsn lenet5 \
    --run_name lenet5_bmrsn \
    --dataset cifar10 \
    --seed 1000 \
    --network lenet5 \
    --batch_size 64 \
    --compress_epochs 1 \
    --kl_averaging mean \
    --learning_rate 0.001026894323862342 \
    --n_epochs 50 \
    --pruning_class LogUniformApproximateDiracBMRPruningLayerpython
```
