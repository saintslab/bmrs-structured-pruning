import torch
import torchvision
import argparse
import numpy as np
import random
from tqdm import tqdm
from carbontracker.tracker import CarbonTracker
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler, Subset
from torch.optim import Adam, AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers.modeling_outputs import ImageClassifierOutput
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import json
from scipy.stats import pearsonr
from collections import defaultdict

from modules.layers import LogUniformAllBMRPruningLayer
from modules.layers import (
    gather_pruning_layers,
    calculate_model_kl,
    usable_neuron_pct,
    compress,
    get_weight_l2,
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


def get_ranks(model, pruning_type='SNR'):
    layers = gather_pruning_layers(model)

    if pruning_type == 'snr':
        return np.argsort(np.argsort(np.concatenate([l.SNR().view(-1).detach().cpu().numpy() for l in layers])))
    elif pruning_type == 'magnitude':
        Qs = get_weight_l2(model)
        return np.argsort(np.argsort(np.concatenate([q.detach().cpu().numpy().reshape(-1) for q in Qs])))
    elif pruning_type == 'bmr_approx':
        return np.argsort(np.argsort(np.concatenate([l.bmr_approximate()[0].view(-1).detach().cpu().numpy() for l in layers]))[::-1])
    elif pruning_type == 'bmr_cdf':
        return np.argsort(np.argsort(np.clip(np.concatenate(
        [l.bmr_cdf(mu_p_tilde=torch.tensor(8).cuda())[0].view(-1).detach().cpu().numpy() for l in layers]),
                       a_min=-1e12, a_max=None))[::-1])
    elif pruning_type == 'bmr_cdf_4':
        return np.argsort(np.argsort(np.clip(np.concatenate(
        [l.bmr_cdf(mu_p_tilde=torch.tensor(4).cuda())[0].view(-1).detach().cpu().numpy() for l in layers]),
                       a_min=-1e12, a_max=None))[::-1])


def generate_heatmap(arrays, names, ax, title=None):
    # Create the grid
    heatmap = []
    for j in range(len(arrays[0])):
        heatmap.append([])
        for array1 in arrays:
            heatmap[-1].append([])
            for array2 in arrays:
                heatmap[-1][-1].append(pearsonr(array1[j], array2[j])[0])
    heatmap = np.array(heatmap).mean(0)

    ax = sns.heatmap(heatmap, ax=ax, cmap='YlGnBu', annot=True,
        annot_kws={"fontsize":26}, cbar=False)
    ax.set_yticks(ax.get_yticks(), labels=names, rotation='horizontal')
    ax.set_xticks(ax.get_xticks(), labels=names, rotation='vertical')
    ax.set_title(title)
    return heatmap


def average_lists(lists):
    # Determine the length of the longest list
    max_length = max(len(lst) for lst in lists)

    # Pad shorter lists with None (or any other placeholder value)
    padded_lists = [lst + [None] * (max_length - len(lst)) for lst in lists]

    # Zip the padded lists
    zipped_lists = zip(*padded_lists)

    # Calculate the average for each index
    averages = [sum([v for v in values if v is not None]) / len([v for v in values if v is not None]) if any(v is not None for v in values) else None
                for values in zip(*padded_lists)]

    stds = [np.std([v for v in values if v is not None]) if any(v is not None for v in values) else None
                for values in zip(*padded_lists)]

    return np.array(averages), np.array(stds)


def get_true_param_comp(model, path, pruning_type, mu_p_tilde, original_comps):
    # Go through the original lambdas, set the mask, calculate the true param comp
    lambdas = np.linspace(0, 100, num=100)
    true_comps = []
    remeasured_neuron_comps = []
    prev_comp = -1
    model.eval()
    for lam in lambdas:
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

    acc_dict = defaultdict(list)
    comp_dict = defaultdict(list)
    rank_dict = defaultdict(list)
    type_to_name = {
        'snr': 'SNR',
        'magnitude': 'L2',
        'bmr_approx': 'BMRS_N',
        'bmr_cdf': 'BMRS_U-8',
        'bmr_cdf_4': 'BMRS_U-4'
    }
    pruning_types = ['snr', 'magnitude', 'bmr_approx', 'bmr_cdf', 'bmr_cdf_4']
    mu_p_tilde = None
    seeds = [100, 1000, 1001, 2001, 50, 5000, 66, 666, 7, 744]
    for seed in seeds:
        model.load_state_dict(torch.load(f'{args.output_dir}/{seed}/cva_{dataset}_{network}_{seed}_snr.pth'))
        for pruning_type in pruning_types:
            if pruning_type == 'bmr_cdf':
                mu_p_tilde = 8
            elif pruning_type == 'bmr_cdf_4':
                mu_p_tilde = 4
            else:
                mu_p_tilde = None
            #if os.path.exists(f"{args.metrics_dir}/{seed}/{dataset}_{network}_{seed}_{pruning_type}.json"):
            with open(f"{args.metrics_dir}/{seed}/{dataset}_{network}_{seed}_{pruning_type}.json") as f:
                metrics = json.loads(f.read())

            if 'true_param_comp' not in metrics:
                metrics['true_param_comp'] = get_true_param_comp(
                    model,
                    f'{args.output_dir}/{seed}/cva_{dataset}_{network}_{seed}_snr.pth',
                    pruning_type,
                    mu_p_tilde,
                    metrics['comps']
                )
            acc_dict[pruning_type].append(metrics['accs'])
            comp_dict[pruning_type].append(metrics['true_param_comp'])
            rank_dict[pruning_type].append(get_ranks(model, pruning_type))

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
    markers = ['o', 's', '*']
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.despine()
    for field in ['magnitude', 'snr']:
        comp_mean,_ = average_lists(comp_dict[field])
        acc_mean, acc_std = average_lists(acc_dict[field])

        sns.lineplot(x=comp_mean, y=acc_mean, ax=ax, label=type_to_name[field], linewidth=6)
        col = ax.lines[-1].get_color()
        colors.append(col)
        plt.fill_between(comp_mean, acc_mean - acc_std, acc_mean + acc_std, color=col, alpha=0.2)

    for j,field in enumerate(['bmr_approx', 'bmr_cdf', 'bmr_cdf_4']):
        comp_mean,_ = average_lists(comp_dict[field])#np.vstack(comp_dict[field]).mean(0)
        max_comps = np.array([c[-1] for c in comp_dict[field]])
        max_accs = np.array([a[-1] for a in acc_dict[field]])

        acc_mean, acc_std = average_lists(acc_dict[field])

        sns.lineplot(x=comp_mean, y=acc_mean, ax=ax, label=type_to_name[field], linewidth=8)
        col = ax.lines[-1].get_color()
        colors.append(col)
        plt.fill_between(comp_mean, acc_mean - acc_std, acc_mean + acc_std, color=col, alpha=0.2)


        plt.vlines(x=max_comps.max(), ymin=max_accs[max_comps.argmax()] - 0.05,
                   ymax=max_accs[max_comps.argmax()] + 0.05, color='black',
                   linewidth=6, zorder=10)
        plt.scatter(x=max_comps.max(), y=max_accs[max_comps.argmax()], color=colors[j+2], s=markersize, zorder=11,
                    linewidth=3, edgecolors='black', marker=markers[j])




    ax.set_xlabel("Compression %")
    ax.set_ylabel(f"Accuracy")
    ax.legend(loc='lower left')


    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/{dataset}_{network}_cva_combined.png')
    plt.savefig(f'{args.output_dir}/{dataset}_{network}_cva_combined.pdf')


    # Now the zoomed in lineplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.despine()
    field_comps = {field: [c[-1] for c in comp_dict[field]] for field in ['bmr_approx', 'bmr_cdf', 'bmr_cdf_4']}
    field_accs = {field: [a[-1] for a in acc_dict[field]] for field in ['bmr_approx', 'bmr_cdf', 'bmr_cdf_4']}

    xmin = min([c for field in field_comps for c in field_comps[field]]) - 5
    xmax = max([c for field in field_comps for c in field_comps[field]]) + 5

    snr_comp_avg, snr_comp_std = average_lists(comp_dict['snr'])
    snr_acc_avg, snr_acc_std = average_lists(acc_dict['snr'])

    snr_comp_acc = [(comp, acc) for comp, acc in zip(snr_comp_avg, snr_acc_avg) if comp >= xmin and comp <= xmax]

    ymin = min([a for field in field_accs for a in field_accs[field]] + [s[1] for s in snr_comp_acc]) - 0.1
    ymax = max([a for field in field_accs for a in field_accs[field]] + [s[1] for s in snr_comp_acc]) + 0.1


    sns.lineplot(x=[s[0] for s in snr_comp_acc], y=[s[1] for s in snr_comp_acc], ax=ax, label=type_to_name['snr'],
                 linewidth=linewidth, color=colors[1], alpha=0.8)

    for j,field in enumerate(['bmr_approx', 'bmr_cdf', 'bmr_cdf_4']):
        plt.scatter(x=field_comps[field], y=field_accs[field], color=colors[j+2], s=markersize, zorder=11,
                    linewidth=2, label=type_to_name[field], edgecolors='black', marker=markers[j])
        sns.kdeplot(
            x=field_comps[field],
            y=field_accs[field],
            levels=5,
            fill=True,
            alpha=0.3,
            cut=2,
            ax=ax,
            color=colors[j+2]
        )

    ax.set_xlabel("Compression %")
    ax.set_ylabel(f"Accuracy")
    ax.set_ylim((ymin, ymax))
    ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/{dataset}_{network}_cva_zoomed_combined.png')
    plt.savefig(f'{args.output_dir}/{dataset}_{network}_cva_zoomed_combined.pdf')

    plt.rc('axes', labelsize=36)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=34)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=34)  # fontsize of the tick labels
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    arrs = [rank_dict[n] for n in pruning_types]
    names = [type_to_name[n] for n in pruning_types]
    heatmap = generate_heatmap(
        arrays=arrs,
        names=names,
        ax=ax,
        title=None
    )
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{dataset}_{network}_cva_correlation_combined.png")
    plt.savefig(f"{args.output_dir}/{dataset}_{network}_cva_correlation_combined.pdf")


