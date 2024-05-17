from torch import nn
import math

from modules.layers import LogUniformPruningLayer

# Define a basic network, add dropout to one layer
def make_mlp(
            in_dim=784,
            out_dim=5,
            hidden_dim=10,
            n_layers=5,
            pruning_class=LogUniformPruningLayer,
            enable_pruning=True
    ):
    """
    Creates an MLP
    :param in_dim: The input dimensionality
    :param out_dim: The number of classes
    :param hidden_dim: The number of neurons in each hidden layer
    :param n_layers: The number of hidden layers
    :param pruning_class: The class to use for BMRS pruning
    :param enable_pruning: Whether or not to enable pruning layers
    :return: An MLP
    """

    linear_blocks = [
        nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            pruning_class(d=[hidden_dim], enabled=enable_pruning)
        )
    for _ in range(n_layers)]

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.Tanh(),
        pruning_class(d=[hidden_dim], enabled=enable_pruning),
        *linear_blocks,
        nn.Linear(hidden_dim, out_dim)
    )

def lenet5(
        pruning_class=LogUniformPruningLayer,
        enable_pruning=True,
        input_channels=1
):
    """
    Create Lenet5; assumes a 32x32 image size
    :param pruning_class: Which BMRS pruning class to use
    :param enable_pruning: Whether or not to enable BMRS pruning
    :param input_channels: The number of image channels
    :return: Lenet5
    """
    return nn.Sequential(
        nn.Conv2d(input_channels, 6*input_channels, 5),
        nn.ReLU(),
        pruning_class(d=[6*input_channels,28,28], enabled=enable_pruning, axis=0),
        nn.MaxPool2d(2),
        nn.Conv2d(6*input_channels, 16*input_channels, 5),
        nn.ReLU(),
        pruning_class(d=[16*input_channels,10,10], enabled=enable_pruning, axis=0),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(400*input_channels, 120*input_channels),
        nn.ReLU(),
        pruning_class(d=[120*input_channels], enabled=enable_pruning, axis=0),
        nn.Linear(120*input_channels, 84*input_channels),
        nn.ReLU(),
        pruning_class(d=[84*input_channels], enabled=enable_pruning, axis=0),
        nn.Linear(84*input_channels, 10)
    )


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
                                              num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(
        dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = math.floor((h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    return h, w


def insert_pruning_resnet(
        model,
        input_size=(224, 224),
        channels=3,
        pruning_class=LogUniformPruningLayer,
        enable_pruning=True,
        recalc_size=True
):
    """
    Recursive function which inserts BMRS pruning layers into a Resnet architecture at the output nodes
    :param model: A Resnet
    :param input_size: The input image size
    :param channels: Number of image channels
    :param pruning_class: The BMRS pruning class to use
    :param enable_pruning: Whether or not to enable BMRS pruning
    :param recalc_size: Whether or not to recalculate the current input size (for convolutional layers)
    :return: A Resnet with pruning layers inserted
    """
    current_size = input_size
    for child_name, child in model.named_children():
        if recalc_size:
            if isinstance(child, nn.Conv2d):
                current_size = conv2d_output_shape(current_size, child.kernel_size, child.stride, child.padding, child.dilation)
                channels = child.out_channels
            elif isinstance(child, nn.MaxPool2d):
                current_size = conv2d_output_shape(current_size, child.kernel_size, child.stride, child.padding, child.dilation)

        if isinstance(child, nn.BatchNorm2d):
            # Add dropout layer
            setattr(model, child_name, nn.Sequential(
                child,
                #nn.ReLU(), # TODO: need to see how to put pruning after non-linearity
                pruning_class(d=[channels, current_size[0], current_size[1]], enabled=enable_pruning, axis=0)
            ))
        else:
            current_size,channels = insert_pruning_resnet(child, current_size, channels, pruning_class, enable_pruning, recalc_size=child_name != 'downsample')
    return current_size,channels


def insert_pruning_vit(
        model,
        pruning_class=LogUniformPruningLayer,
        enable_pruning=True,
        parent='None'
):
    """
    Recursive function to insert BMRS pruning layers into a vision transformer
    :param model: A vision transformer model
    :param pruning_class: The BMRS pruning class to use
    :param enable_pruning: Whether or not to enable BMRS pruning
    :param parent: The name of the parent layer (to find output layers)
    :return: A vision transformer with pruning layers inserted
    """
    for child_name, child in model.named_children():
        #if parent == 'attention' or parent == 'output':
        if parent == 'output':
            if isinstance(child, nn.Linear):
                # Add dropout layer
                setattr(model, child_name, nn.Sequential(
                    child,
                    #nn.ReLU(),
                    pruning_class(d=[197,768], enabled=enable_pruning, axis=1)
                ))
            else:
                insert_pruning_vit(child, pruning_class, enable_pruning, child_name)
        else:
            insert_pruning_vit(child, pruning_class, enable_pruning, child_name)
