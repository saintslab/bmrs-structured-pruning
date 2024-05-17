import torch
import torchvision
import argparse
import numpy as np
import random
from tqdm import tqdm
from carbontracker.tracker import CarbonTracker
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler, Subset
from torch.optim import Adam, SGD, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers.modeling_outputs import ImageClassifierOutput
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import ipdb
import os
import json
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

from modules.layers import LogUniformAllBMRPruningLayer
from modules.layers import (
    gather_pruning_layers,
    calculate_model_kl,
    usable_neuron_pct,
    compress,
    get_true_parameter_compression_percent,
    get_weight_l2,
    enable,
    disable,
    get_true_parameter_compression_percent
)
from modules.classifiers import make_mlp, lenet5, insert_pruning_resnet, insert_pruning_vit


def enforce_reproducibility(seed=1000):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


def set_masks(model, pct, criterion='snr', mu_p_tilde=None):
    layers = gather_pruning_layers(model)
    if criterion == 'snr':
        Qs = [l.SNR() for l in layers]
        threshold = np.percentile(np.concatenate([q.detach().cpu().numpy().reshape(-1) for q in Qs]), pct)
    elif criterion == 'magnitude':
        Qs = get_weight_l2(model)#[l.Etheta() for l in layers]
        threshold = np.percentile(np.concatenate([q.detach().cpu().numpy().reshape(-1) for q in Qs]), pct)
    elif criterion == 'bmr_approx':
        Qs = [l.bmr_approximate()[0] for l in layers]
        threshold = np.percentile(np.concatenate([q.detach().cpu().numpy().reshape(-1) for q in Qs]), 100 - pct)
    elif criterion == 'bmr_exact':
        Qs = [torch.clamp(l.bmr_exact(mu_p_tilde=torch.tensor(np.log(1/8)).cuda())[0], min=-1e12, max=None) for l in layers]
        threshold = np.percentile(np.concatenate([q.detach().cpu().numpy().reshape(-1) for q in Qs]), 100 - pct)

    elif criterion == 'bmr_cdf':
        mu_p_tilde = torch.tensor(8).double().to(device)
        Qs = [torch.clamp(l.bmr_cdf(mu_p_tilde=mu_p_tilde)[0], min=-1e12, max=None) for l in layers]
        threshold = np.percentile(np.concatenate([q.detach().cpu().numpy().reshape(-1) for q in Qs]), 100 - pct)

    elif criterion == 'bmr_cdf_4':
        mu_p_tilde = torch.tensor(4).double().to(device)
        Qs = [torch.clamp(l.bmr_cdf(mu_p_tilde=mu_p_tilde)[0], min=-1e12, max=None) for l in layers]
        threshold = np.percentile(np.concatenate([q.detach().cpu().numpy().reshape(-1) for q in Qs]), 100 - pct)


    for q,l in zip(Qs,layers):
        if criterion in ['snr', 'magnitude']:
            l.mask.data = (q > threshold).float().view(l.mask.shape)
        elif 'bmr' in criterion:
            l.mask.data = ((q < threshold) | (q < 0.)).float()



def get_true_param_comp(model, path, pruning_type, mu_p_tilde, original_comps):
    # Go through the original lambdas, set the mask, calculate the true param comp
    lambdas = np.linspace(0, 100, num=100)
    # sigs = np.logspace(-6,0,100)
    true_comps = []
    remeasured_neuron_comps = []
    prev_comp = -1
    model.eval()
    for lam in lambdas:
        # for sig in sigs:
        # for p in ps:
        model.load_state_dict(torch.load(path))

        set_masks(model, lam, pruning_type, mu_p_tilde)
        comp = usable_neuron_pct(model)
        if comp != prev_comp:
            true_comps.append(get_true_parameter_compression_percent(model).item()*100)
            remeasured_neuron_comps.append(comp)
            prev_comp = comp
    assert all([rc == c for rc,c in zip(remeasured_neuron_comps, original_comps)])
    return true_comps


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", help="A name for this run", default="mnist_mlp_stable_baseline", type=str)
    parser.add_argument("--metrics_dir", help="Directory to save metrics in", required=True, type=str)
    parser.add_argument("--output_dir", help="Directory to save models in", required=True, type=str)

    parser.add_argument("--network", help="Name of the network to use", default='mlp',
                        choices=['mlp', 'lenet5', 'resnet18', 'resnet50', 'vit'], type=str)
    parser.add_argument("--dataset", help="Name of the network to use", default='mnist',
                        choices=['mnist', 'cifar10', 'fashion_mnist'], type=str)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=32)
    parser.add_argument("--n_layers", help="Number of layers", type=int, default=5)
    parser.add_argument("--hidden_dim", help="Number of units in each layer", type=int, default=50)
    parser.add_argument("--learning_rate", help="The learning rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", help="Amount of weight decay", type=float, default=0.0)
    parser.add_argument("--n_epochs", help="The number of epochs to run", type=int, default=20)
    parser.add_argument("--disable_pruning", help="Disables pruning", action="store_true")
    parser.add_argument("--debug", help="Debug mode", action="store_true")
    parser.add_argument("--kl_averaging", help="How to reduce KL divergence", type=str, choices=['sum', 'mean'],
                        default='mean')
    parser.add_argument("--pretrained", help="Whether or not to use a pretrained model", action="store_true")
    parser.add_argument("--tags", help="Tags to pass to wandb", required=False, type=str, default=[], nargs='+')

    args = parser.parse_args()

    # Always first
    enforce_reproducibility(1000)
    if not os.path.exists(f"{args.metrics_dir}"):
        os.makedirs(f"{args.metrics_dir}")
    if not os.path.exists(f"{args.output_dir}"):
        os.makedirs(f"{args.output_dir}")

    epochs = args.n_epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    use_pruning_layers = not args.disable_pruning
    kl_averaging = args.kl_averaging
    pruning_class = LogUniformAllBMRPruningLayer
    n_layers = args.n_layers
    hidden_dim = args.hidden_dim
    network = args.network
    flatten = network == 'mlp'
    weight_decay = args.weight_decay
    dataset = args.dataset
    img_size = 32 if network in ['mlp', 'lenet5'] else 224


    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device('cuda')
    if 'mnist' in dataset:
        input_channels = 1
        n_classes = 10
    elif dataset == 'cifar10':

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        input_channels = 3
        n_classes = len(classes)
    in_dim = 32 * 32 * input_channels if network in ['mlp', 'lenet5'] else 224*224*input_channels
    scheduler = None
    if network == 'mlp':
        model = make_mlp(
            in_dim=in_dim,
            out_dim=n_classes,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            pruning_class=pruning_class,
            enable_pruning=use_pruning_layers
        ).to(device)
    elif network == 'lenet5':
        model = lenet5(pruning_class, enable_pruning=use_pruning_layers, input_channels=input_channels).to(device)
    elif 'resnet' in network:
        model = torch.hub.load('pytorch/vision:v0.10.0', network, pretrained=args.pretrained)
        insert_pruning_resnet(model, pruning_class=pruning_class, enable_pruning=use_pruning_layers)
        model = model.to(device)
    elif network == 'vit':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        insert_pruning_vit(model, pruning_class=pruning_class, enable_pruning=use_pruning_layers)
        model = model.to(device)


    accuracies = []
    comps = []
    mu_p_tildes = [16, 12, 8, 6, 4, 2]
    mu_p_tilde = None
    seeds = [1000, 1001, 666]
    for mu_p_tilde in mu_p_tildes:
        base_dir = f"{dataset}_bmr_cdf_multiple_{mu_p_tilde}_{network}"
        accuracies.append([])
        comps.append([])
        for seed in seeds:
            model.load_state_dict(torch.load(f'{args.output_dir}/{base_dir}/{dataset}_{network}_{seed}.pth'))

            #if os.path.exists(f"{args.metrics_dir}/{seed}/{dataset}_{network}_{seed}_{pruning_type}.json"):
            with open(f"{args.metrics_dir}/{base_dir}/{seed}.json") as f:
                metrics = json.loads(f.read())

            if 'true_param_comp' not in metrics:
                metrics['true_param_comp'] = get_true_parameter_compression_percent(model, vit=model[0] == 'vit')
            accuracies[-1].append(metrics['test_acc'] * 100)
            #comp_dict[pruning_type].append(metrics['comps'])
            comps[-1].append(metrics['true_param_comp'] * 100)

    #with sns.plotting_context(font_scale=1.):
    #sns.set(font_scale=2.5)
    plt.rc('axes', titlesize=36)  # fontsize of the axes title
    plt.rc('axes', labelsize=40)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=38)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=38)  # fontsize of the tick labels
    plt.rc('legend', fontsize=34)  # legend fontsize
    sns.set_style("white")
    sns.set_palette(sns.color_palette("colorblind"))
    colors = []
    linewidth = 12
    markersize = 400
    # Scatter plots for the others
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sns.despine()

    scatter_x = [mu_p_tildes[i] for i in range(len(accuracies)) for j in range(len(accuracies[i]))]
    scatter_y_acc = [acc for mu in accuracies for acc in mu]
    p1 = sns.pointplot(x=scatter_x, y=scatter_y_acc, errorbar='ci', capsize=0.1, ax=ax, color=sns.color_palette("colorblind")[0], label="Accuracy", linewidth=6)#, linewidth=linewidth, markersize=markersize)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("$p_{1}$")

    ax2 = ax.twinx()

    scatter_x = [mu_p_tildes[i] for i in range(len(accuracies)) for j in range(len(accuracies[i]))]
    scatter_y_comp = [comp for mu in comps for comp in mu]
    p2 = sns.pointplot(x=scatter_x, y=scatter_y_comp, errorbar='ci', capsize=0.1, ax=ax2, color=sns.color_palette("colorblind")[1], label="Compression %", linewidth=6)
    ax2.set_ylabel("Compression %")
    #ax.set_xlabel("$$p_{1}$$")
    plt.tight_layout()

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower left')
    ax.get_legend().remove()
    plt.savefig(f"latex/figures/{dataset}_{network}_bmrsu.png")
    plt.savefig(f"latex/figures/{dataset}_{network}_bmrsu.png")


