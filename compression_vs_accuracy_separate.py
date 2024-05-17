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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import json
from scipy.stats import pearsonr

from modules.layers import LogUniformAllBMRPruningLayer
from modules.layers import (
    gather_pruning_layers,
    usable_neuron_pct,
    compress,
    get_weight_l2,
    calculate_model_kl,
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


def process_data(data, batch_size=32, shuffle=False, flatten=True, resize=32):
    X = []
    Y = []
    resize = T.Resize(resize, antialias=True)
    for d in tqdm(data):
        if flatten:
            X.append(resize(torch.tensor(np.array(d['image'])).unsqueeze(0)).view(-1))
        else:
            X.append(resize(torch.tensor(np.array(d['image'])).unsqueeze(0)))
        Y.append(d['label'])
    X = (torch.stack(X).to(torch.float32) - 127) / 255
    Y = torch.LongTensor(Y)
    dset = TensorDataset(X, Y)
    loader = DataLoader(dset, shuffle=shuffle, batch_size=batch_size, num_workers=0)

    return X,Y,dset,loader


def evaluate(model, loader, nll=False):
    all_preds = []
    all_top5 = []
    all_labs = []
    all_nll = []
    model.eval()
    with torch.no_grad():
        for X, Y in tqdm(loader):
            X = X.to(device)
            Y = Y.to(device)
            logits = model(X)
            if isinstance(logits, ImageClassifierOutput):
                logits = logits['logits']
            all_preds.extend(np.argmax(logits.detach().cpu().numpy(), -1))
            all_top5.extend([np.argsort(l,-1)[::-1][:5] for l in logits.detach().cpu().numpy()])
            all_labs.extend(np.array(Y.detach().cpu()))
            all_nll.append(F.nll_loss(F.log_softmax(logits, -1), Y).sum().item())

    if nll:
        return (np.vstack(all_preds) == np.vstack(all_labs)).mean(), np.array([lab in top5 for lab,top5 in zip(all_labs, all_top5)]).mean(), np.array(all_nll).sum()
    else:
        return (np.vstack(all_preds) == np.vstack(all_labs)).mean(), np.array([lab in top5 for lab,top5 in zip(all_labs, all_top5)]).mean()


def train_loop(model, loader, optim, n_epochs, criterion, tracker, compress_epochs=1, kl_averaging='sum', tqdm_desc="Training", output_dir='models', scheduler=None, save_best=True):
    best_acc = 0.0
    best_top5 = 0.0
    with tqdm(range(n_epochs), desc=tqdm_desc) as pbar:
        for e in pbar:
            tracker.epoch_start()
            model.train()
            for X, Y in tqdm(loader):
                X = X.to(device)
                Y = Y.to(device)
                optim.zero_grad()
                logits = model(X)
                if isinstance(logits, ImageClassifierOutput):
                    logits = logits['logits']
                recon = criterion(logits, Y)
                kl_loss = calculate_model_kl(model, averaging=kl_averaging)
                loss = recon + kl_loss
                loss.backward()
                optim.step()
                if scheduler:
                    scheduler.step()

            tracker.epoch_end()
            # Prune now for some models
            #comp_pct = model.compress(compress=compress_epochs > 0 and (((e + 1) % compress_epochs) == 0))
            # if e >= 39:
            #     ipdb.set_trace()
            comp_pct = compress(model, set_mask=compress_epochs > 0 and (((e + 1) % compress_epochs) == 0))
            true_param_comp = get_true_parameter_compression_percent(model)
            acc,top5_acc,_ = evaluate(model, val_loader, nll=True)
            print(comp_pct, true_param_comp, acc, top5_acc)

            #wandb.log({"Compression %": comp_pct, 'val_acc': acc, 'val_top5_acc': top5_acc, 'true_param_comp': true_param_comp})
            if save_best:
                torch.save(model.state_dict(), f'{args.output_dir}/{dataset}_{network}_{seed}.pth')
            if acc > best_acc:
                #torch.save(model.state_dict(), f'{args.output_dir}/{dataset}_{network}_{seed}.pth')
                best_acc = acc
            best_top5 = max(best_top5, top5_acc)
            pbar.set_description(f"[{tqdm_desc} iteration %04d] loss: %.4f" % (e + 1, loss / len(X)))
    tracker.stop()

    return model, best_acc, best_top5


def get_transforms(network, flatten=False):
    if flatten:
        fl = T.Lambda(lambda x: torch.flatten(x))
    else:
        fl = T.Lambda(lambda x: x)
    if network in ['lenet5', 'mlp']:
        train_transform = T.Compose(
            [T.ToTensor(),
             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             fl])
        val_transform = train_transform
    elif 'resnet' in network:
        train_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            fl
        ])
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            fl
        ])
    elif network == 'vit':
        feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        normalize = T.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        train_transform = T.Compose(
            [
                T.RandomResizedCrop(tuple(feature_extractor.size.values())),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
                fl
            ]
        )
        val_transform = T.Compose(
            [
                T.Resize(tuple(feature_extractor.size.values())),
                T.CenterCrop(tuple(feature_extractor.size.values())),
                T.ToTensor(),
                normalize,
                fl
            ]
        )


    return train_transform, val_transform


def set_masks(model, pct, criterion='snr'):
    layers = gather_pruning_layers(model)
    if criterion == 'snr':
        Qs = [l.SNR() for l in layers]
        threshold = np.percentile(np.concatenate([q.detach().cpu().numpy().reshape(-1) for q in Qs]), pct)
    elif criterion == 'magnitude':
        Qs = get_weight_l2(model)
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


def generate_heatmap(arrays, names, ax, title=None):
    # Create the grid
    heatmap = []
    for array1 in arrays:
        heatmap.append([])
        for array2 in arrays:
            heatmap[-1].append(pearsonr(array1, array2)[0])

    ax = sns.heatmap(heatmap, cbar=True, ax=ax, cmap='YlGnBu', annot=True)
    ax.set_yticks(ax.get_yticks(), labels=names, rotation='horizontal')
    ax.set_xticks(ax.get_xticks(), labels=names, rotation='vertical')
    ax.set_title(title)
    return heatmap


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", help="A name for this run", default="mnist_mlp_stable_baseline", type=str)
    parser.add_argument("--metrics_dir", help="Directory to save metrics in", required=True, type=str)
    parser.add_argument("--output_dir", help="Directory to save models in", required=True, type=str)
    parser.add_argument("--type", help="The type of pruning to test", required=True, type=str, choices=['snr', 'magnitude', 'bmr_approx', 'bmr_cdf', 'bmr_cdf_4'])

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
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--disable_pruning", help="Disables pruning", action="store_true")
    parser.add_argument("--debug", help="Debug mode", action="store_true")
    parser.add_argument("--kl_averaging", help="How to reduce KL divergence", type=str, choices=['sum', 'mean'],
                        default='mean')
    parser.add_argument("--pretrained", help="Whether or not to use a pretrained model", action="store_true")
    parser.add_argument("--tags", help="Tags to pass to wandb", required=False, type=str, default=[], nargs='+')
    parser.add_argument("--mu_p_tilde", help="mu to use for new prior", type=float, default=None)

    args = parser.parse_args()

    # Always first
    seed = args.seed
    enforce_reproducibility(seed)
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
    pruning_type = args.type
    mu_p_tilde = args.mu_p_tilde


    device = torch.device('cpu')
    if torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device('cuda')

    if mu_p_tilde != None:
        mu_p_tilde = torch.tensor(mu_p_tilde).double().to(device)

    if 'mnist' in dataset:
        mnist = load_dataset(dataset)

        train_data = [d for d in mnist['train']]
        train_data, val_data = train_test_split(train_data, test_size=0.2)
        Xtrain, Ytrain, train_dset, train_loader = process_data(train_data, shuffle=True, batch_size=batch_size, flatten=flatten)
        ft_loader = DataLoader(train_dset, shuffle=False, batch_size=batch_size, num_workers=0)
        Xval, Yval, val_dset, val_loader = process_data(val_data, batch_size=batch_size, flatten=flatten)

        test_data = [d for d in mnist['test']]
        Xtest, Ytest, test_dset, test_loader = process_data(test_data, batch_size=batch_size, flatten=flatten)
        input_channels = 1
        n_classes = 10
    elif dataset == 'cifar10':

        train_transform, val_transform = get_transforms(network, flatten)

        trainset = torchvision.datasets.CIFAR10(root='./cache', train=True,
                                                download=True, transform=train_transform)
        valset = torchvision.datasets.CIFAR10(root='./cache', train=True,
                                              download=True, transform=val_transform)
        idx = list(range(len(trainset)))
        train_idx, val_idx = train_test_split(idx, test_size=0.2)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   num_workers=0, sampler=SubsetRandomSampler(indices=train_idx))
        ft_loader = torch.utils.data.DataLoader(Subset(trainset, train_idx), batch_size=batch_size,
                                                   num_workers=0, shuffle=False)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                 num_workers=0,
                                                 sampler=SubsetRandomSampler(indices=val_idx))

        testset = torchvision.datasets.CIFAR10(root='./cache', train=False,
                                               download=True, transform=val_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=0)

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
        optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif network == 'lenet5':
        model = lenet5(pruning_class, enable_pruning=use_pruning_layers, input_channels=input_channels).to(device)
        optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif 'resnet' in network:
        model = torch.hub.load('pytorch/vision:v0.10.0', network, pretrained=args.pretrained)
        insert_pruning_resnet(model, pruning_class=pruning_class, enable_pruning=use_pruning_layers)
        model = model.to(device)
        optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif network == 'vit':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        insert_pruning_vit(model, pruning_class=pruning_class, enable_pruning=use_pruning_layers)
        model = model.to(device)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optim = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optim,
            200,
            epochs * len(train_loader)  # total
        )


    tracker = CarbonTracker(epochs=epochs, log_dir='./carbontracker')

    metrics = {}

    criterion = CrossEntropyLoss()

    if not os.path.exists(f'{args.output_dir}/cva_{dataset}_{network}_{seed}_{pruning_type}.pth'):
        model,val_acc,top5_acc = train_loop(model, train_loader, optim, epochs, criterion, tracker, compress_epochs=-1, kl_averaging=kl_averaging, tqdm_desc="Train", output_dir=args.output_dir, scheduler=scheduler)

        torch.save(model.state_dict(), f'{args.output_dir}/cva_{dataset}_{network}_{seed}_{pruning_type}.pth')

    model.load_state_dict(torch.load(f'{args.output_dir}/cva_{dataset}_{network}_{seed}_{pruning_type}.pth'))
    acc = evaluate(model, test_loader)

    lambdas = np.linspace(0, 100, num=100)
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    accs = []
    comps = []

    model.eval()

    for j,lam in enumerate(lambdas):
        model.load_state_dict(torch.load(f'{args.output_dir}/cva_{dataset}_{network}_{seed}_{pruning_type}.pth'))

        set_masks(model, lam, pruning_type, mu_p_tilde)
        train_loop(model, ft_loader, optim, 1, criterion, tracker, compress_epochs=-1, kl_averaging=kl_averaging,
               tqdm_desc="Fine-tune", output_dir=args.output_dir, scheduler=scheduler, save_best=False)
        comp = usable_neuron_pct(model)
        if j == 0 or comp != comps[-1]:
            acc = evaluate(model, test_loader)
            accs.append(acc[0])
            comps.append(comp)


    metrics = {
        'lambda': list(lambdas),
        'comps': comps,
        'accs': accs,

    }
    with open(f"{args.metrics_dir}/{dataset}_{network}_{seed}_{pruning_type}.json", 'wt') as f:
        f.write(json.dumps(metrics))