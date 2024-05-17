import argparse
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import os
import matplotlib


font = {'size'   : 18}

matplotlib.rc('font', **font)

TABLE_1_HEAD = """
\\begin{table}%[htp]
    %\\setlength{\\tabcolsep}{1.5pt}
    \\def\\arraystretch{1.2}
    \\centering
    %\\fontsize{10}{10}\\selectfont
    \\scriptsize
    %\\rowcolors{2}{gray!10}{white}
    \\begin{tabular}{l c c | c c | c c}
    \\toprule %\\thickhline
    & \\multicolumn{2}{c}{ {d0} } & \\multicolumn{2}{c}{ {d1} } & \\multicolumn{2}{c}{ {d2} } \\\\
    \\midrule
    Pruning Method & Comp \\% & Acc. & Comp \\% & Acc. & Comp \\% & Acc.\\\\
"""

TABLE_1_FOOT = """
    \\bottomrule % \\thickhline

    \\end{tabular}
    \\caption{Parameter compression \\% and accuracy. Results are averaged across 10 runs. Standard deviation given in subscript.} %Results are reported as the average macro F1 over 5 random seeds.}
    \\label{tab:small_results}
\\end{table}
"""

TABLE_2_HEAD = """
\\begin{table}%[htp]
    %\\setlength{\\tabcolsep}{1.5pt}
    \\def\\arraystretch{1.2}
    \\centering
    %\\fontsize{10}{10}\\selectfont
    \\scriptsize
    %\\rowcolors{2}{gray!10}{white}
    \\begin{tabular}{l c c | c c}
    \\toprule %\\thickhline
    & \\multicolumn{2}{c}{ {d0} } & \\multicolumn{2}{c}{ {d1} } \\\\
    \\midrule
    Pruning Method & Comp \\% & Acc. & Comp \\% & Acc.\\\\
"""

TABLE_2_FOOT = """
    \\bottomrule % \\thickhline

    \\end{tabular}
    \\caption{Parameter compression \\% and accuracy. Results are averaged across 3 runs. Standard deviation given in subscript.} %Results are reported as the average macro F1 over 5 random seeds.}
    \\label{tab:large_results}
\\end{table}
"""


CORRELATION_TABLE_HEAD = """
\\begin{table}%[htp]
    %\\setlength{\\tabcolsep}{1.5pt}
    \\def\\arraystretch{1.2}
    \\centering
    \\fontsize{10}{10}\\selectfont
    \\rowcolors{2}{gray!10}{white}
    \\begin{tabular}{l c c c c}
    \\toprule %\\thickhline
    Dataset & Avg & Centroid & Temp & Hybrid \\\\
    \\midrule 
"""

CORRELATION_TABLE_FOOT = """
    \\bottomrule % \\thickhline

    \\end{tabular}
    \\caption{Pearon correlation between the JS divergence of different aggregation methods and individual methods correlated with the raw performance of the individual methods ($r_{p}$) and the individual method's average JS divergence to each other individual method ($r_{\mu}$). **indicates significance at p < 0.05.} %Results are reported as the average macro F1 over 5 random seeds.}
    \\label{tab:correlation_results}
\\end{table}
"""


def format_string(mu, std, metric, method):
    if mu[method[0]][metric] >= 0:
        return f"{mu[method[0]][metric]:.2f} \\pm {std[method[0]][metric]:.2f}"
    else:
        return f"-"



def write_table(datasets, methods, models, metric_map, table_head, table_foot, outfile_name, metrics_dir):
    seed_count = defaultdict(int)
    # method_to_ranks_map = {'F1': defaultdict(list), 'test_NLL_post': defaultdict(list), 'test_Brier_post': defaultdict(list)}
    # all_score_map = {'F1': defaultdict(list), 'test_NLL_post': defaultdict(list), 'test_Brier_post': defaultdict(list)}
    table_head = table_head
    table_blocks = defaultdict(list)
    for model in models:
        # method_to_ranks_map = {'test_acc': defaultdict(list), 'compression': defaultdict(list),
        #                        }
        # all_score_map = {'test_acc': defaultdict(list), 'compression': defaultdict(list)}
        # rank_map_combined = defaultdict(list)
        for n, dataset in enumerate(datasets):
            table_head = table_head.replace(f"{{d{n}}}", dataset[1])
            # if not os.path.exists(f"{args.output_loc}/{dataset[1]}"):
            #     os.makedirs(f"{args.output_loc}/{dataset[1]}")

            # Iterate through all metrics files for ranking baseline
            metrics_by_baseline = defaultdict(lambda: {'Accuracy': [],
                                                       'Comp %': []})

            for method in methods:
                pth = f"{metrics_dir}/{dataset[0]}_{method[0]}_{model[0]}"
                if dataset[0] == 'cifar10' and 'bmr' in method[0] and model[0] == 'mlp':
                    if method[0] == 'bmr_LogUniformCDFBMRPruningLayer_8':
                        pth = f"{metrics_dir}/{dataset[0]}_bmr_mlp_LogUniformCDFBMRPruningLayer_8"
                    elif method[0] == 'bmr_LogUniformExactDiracBMRPruningLayer_8':
                        pth = f"{metrics_dir}/{dataset[0]}_bmr_mlp_LogUniformExactDiracBMRPruningLayer_8"
                # if dataset[0] == 'cifar10' and 'bmr' in method[0] and model[0] == 'vit':
                #     pth = pth.replace("8", "4")

                for fname in Path(pth).glob('*.json'):
                    seed_count[f"{dataset[1]}_{model[1]}_{method[1]}"] += 1
                    with open(fname) as f:
                        metrics = json.loads(f.read())
                    for m in metric_map:
                        if m in ["compression"]:
                            metrics_by_baseline[method[0]][metric_map[m][0]].append(metrics[m])
                        else:
                            metrics_by_baseline[method[0]][metric_map[m][0]].append(metrics[m] * 100)
                        # plot_dframes[metric_map[m][0]].append([metrics_by_baseline[method[0]][metric_map[m][0]][-1], group_name_map[group], method[1]])

            # Get the means and variances
            mu = defaultdict(dict)
            std = defaultdict(dict)
            metrics_str = ''
            for method in methods:
                for m in metric_map:
                    # if len(metrics_by_baseline[method[0]][metric_map[m][0]]) == 0:
                    #     ipdb.set_trace()
                    mu[method[0]][metric_map[m][0]] = np.array(
                        metrics_by_baseline[method[0]][metric_map[m][0]]).mean() if len(
                        metrics_by_baseline[method[0]][metric_map[m][0]]) > 0 else -1
                    std[method[0]][metric_map[m][0]] = np.array(
                        metrics_by_baseline[method[0]][metric_map[m][0]]).std() if len(
                        metrics_by_baseline[method[0]][metric_map[m][0]]) > 0 else -1
            table_blocks[model[1]].append((mu, std))
        table_string = table_head + '\n'
        table_string += "\\midrule\n"
        method_names = [method[1] for method in methods]
        # Rank the metrics
        table_string_blocks = {}
        for model in table_blocks:
            table_row_strings = [''] * len(methods)
            for mu, std in table_blocks[model]:
                ranks = {}
                for m in metric_map:
                    values = [(method[0], mu[method[0]][metric_map[m][0]]) for method in methods]
                    ranks[m] = list(sorted(values, key=lambda x: x[1], reverse=metric_map[m][1] == "max"))

                for i, method in enumerate(methods):
                    if table_row_strings[i] == '':
                        table_row_strings[i] += f"{method[1]}&"
                    for m in metric_map:
                        # if ranks[m][0][0] == method[0]:
                        #     table_row_strings[i] += f"$\\mathbf{{{format_string(mu, std, metric_map[m][0], method)}}}$& "
                        # # elif ranks[metric_map[m][0]][1][0] == method[0]:
                        # #     table_string += f"$\\underline{{{format_string(mu, std, m, method)}}}$& "
                        # else:
                        #     table_row_strings[i] += f"${format_string(mu, std, metric_map[m][0], method)}$& "
                        table_row_strings[i] += f"${format_string(mu, std, metric_map[m][0], method)}$& "
            table_string_blocks[model] = table_row_strings
    table_string = table_head + '\n'
    for model in table_string_blocks:
        table_row_strings = table_string_blocks[model]
        table_string += f"\\midrule\n\\multicolumn{{{len(datasets)*len(metric_map) + 1}}}{{c}}{{{model}}}\\\\\n\\midrule\n"
        for row_string in table_row_strings:
            table_string += row_string[:-2] + "\\\\\n"

    table_string += table_foot

    with open(outfile_name, 'wt') as f:
        f.write(table_string)

    print(seed_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir",
                        help="The location of the baseline metrics",
                        type=str, required=True)
    parser.add_argument("--output_loc",
                        help="Where to save the tables",
                        type=str, default='latex/')

    args = parser.parse_args()

    if not os.path.exists(f"{args.output_loc}"):
        os.makedirs(f"{args.output_loc}")

    #### Experiments for Table 1######
    datasets = [
        ("mnist", "MNIST"),
        ("fashion_mnist", "Fash-MNIST"),
        ("cifar10", "CIFAR10"),
    ]

    ## TODO: can extend BMR to all model types and precisions
    methods = [
        ('basic', 'None'),
        ('l2_norm', 'L2'),
        ('snr', 'SNR'),
        ('bmr', '\\methodn{}'),
        #('bmr_cdf', 'CDF (1%)'),
        ('bmr_cdf_23_8', '\\methodu{}-8'),
        ('bmr_cdf_23_4', '\\methodu{}-4')
        #('bmr_exact', 'Exact Spike')
    ]

    models = [
        ('mlp', 'MLP'),
        ('lenet5', 'Lenet5'),
    ]

    # metric_map = {
    #     "compression": ("Comp %", "max"),
    #     "test_acc": ("Accuracy", "max")
    # }
    metric_map = {
        'true_parameter_compression_pct': ("Comp %", "max"),
        "test_acc": ("Accuracy", "max")
    }

    write_table(
        datasets,
        methods,
        models,
        metric_map,
        TABLE_1_HEAD,
        TABLE_1_FOOT,
        f"{args.output_loc}/table1.tex",
        args.metrics_dir
    )

    #### Experiments for Table 2######
    datasets = [
        ("cifar10", "CIFAR10"),
        ("tinyimagenet", "TinyImagenet")
    ]

    ## TODO: can extend BMR to all model types and precisions
    methods = [
        ('basic', 'None'),
        ('l2_norm', 'L2'),
        ('snr', 'SNR'),
        ('bmr', '\\methodn{}'),
        #('bmr_cdf', 'CDF (1%)'),
        ('bmr_cdf_23_8', '\\methodu{}-8'),
        ('bmr_cdf_23_4', '\\methodu{}-4')
        # ('bmr_exact', 'Exact Spike')
    ]

    models = [
        # ('resnet50', 'Res50'),
        ('resnet50pretrained', 'Res50-Pretrained'),
        ('vit', 'Vision Transformer')
    ]

    # metric_map = {
    #     "compression": ("Comp %", "max"),
    #     "test_acc": ("Accuracy", "max")
    # }
    metric_map = {
        'true_parameter_compression_pct': ("Comp %", "max"),
        "test_acc": ("Accuracy", "max")
    }

    write_table(
        datasets,
        methods,
        models,
        metric_map,
        TABLE_2_HEAD,
        TABLE_2_FOOT,
        f"{args.output_loc}/table2.tex",
        args.metrics_dir
    )



