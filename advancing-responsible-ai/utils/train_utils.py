from torch import nn, optim
import torch

from .dataset_utils import create_dataloader_from_df
from .models import MLP, ConvNet


def train_baseline_model(df):
    # Use these network size and seed initializations in the baseline model.
    n_epochs = 3
    hidden_dim = 10
    data_shuffle_seed = 0
    model_weight_init_seed = 0

    return train_model(
        df=df,
        img_col="img",
        label_col="label",
        colored=True,
        dataset_name="train_set",
        net_type="MLP",
        n_epochs=n_epochs,
        hidden_dim=hidden_dim,
        data_shuffle_seed=data_shuffle_seed,
        model_weight_init_seed=model_weight_init_seed,
    )


def train_model(
    df,
    img_col,
    label_col,
    colored,
    dataset_name,
    net_type,
    n_epochs,
    hidden_dim,
    data_shuffle_seed,
    model_weight_init_seed,
):
    """
    :param df: dataframe of training data
    :param img_col: column in df for the img
    :param label_col: column in df for the label
    :param colored: boolean denoting if the images are colored. 3 input channels vs. 1
    :param dataset_name: name of the dataset
    :param net_type: 'MLP' is only supported option
    :param n_epochs: number of training epochs
    :param data_shuffle_seed: random seed used before we initialize the dataloader
    :param hidden_dim: Integer dimension to use for the hidden layers of MLP
    :param model_weight_init_seed: Random seed used to initialize model weights with pytorch's
        `manual_seed`
    :return:
    """
    train_loader = create_dataloader_from_df(
        df, img_col, label_col, dataset_name, data_shuffle_seed
    )
    model = train_model_from_loader(
        train_loader, colored, net_type, n_epochs, hidden_dim, model_weight_init_seed
    )
    return model


def train_model_from_loader(
    train_loader, colored, net_type, n_epochs, hidden_dim, model_weight_init_seed
):
    """
    :param train_loader: An pytorch train loader for a VisionDataset
    :param colored: Bool denoting if the dataset is colored, affects input channels of neural net
    :param net_type: Either 'Conv' or 'MLP'
    :param n_epochs: number of epochs to train for
    :param train_loader: Pytorch dataloader of training data
    :param hidden_dim: Integer dimension to use for the hidden layers of MLP
    :param model_weight_init_seed: Random seed used to initialize model weights with pytorch's
        `manual_seed`
    :return:
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Define the model
    if net_type == "MLP":
        model = MLP(
            model_weight_init_seed=model_weight_init_seed,
            colored=colored,
            hidden_dim=hidden_dim,
        ).to(device)
    elif net_type == "Conv":
        model = ConvNet(
            model_weight_init_seed=model_weight_init_seed, colored=colored
        ).to(device)
    else:
        raise RuntimeError("Invalid network type")

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(1, n_epochs + 1):
        train_epoch(model, device, train_loader, optimizer, epoch)

    return model


def train_epoch(model, device, train_loader, optimizer, epoch, verbose=False):
    """
    Trains a model for one epoch on the dataset.

    :param model:
    :param device:
    :param train_loader:
    :param optimizer:
    :param epoch:
    :param verbose:
    :return:
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (data, target, *_) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if verbose:
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
