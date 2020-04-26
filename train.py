"""
Author: Can Bakiskan
Date: 2019-06-07

Trains specified model (base or blackbox), with appropriate options,
and saves it to specified path.

"""

import os
import argparse
from tqdm import tqdm
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST
from models import *
import plot_settings
import matplotlib.pyplot as plt
import csv
from parameters import get_arguments
from types import BuiltinFunctionType
from plot_settings import cm

model_dict = {
    "Classifier": Classifier,
    "Classifier_no_maxpool": Classifier_no_maxpool,
    "Direct_quantization_model": Direct_quantization_model,
    "Polarization_quantization_model": Polarization_quantization_model,
    "Polarization_quantization_model_no_maxpool": Polarization_quantization_model_no_maxpool,
}


def plot_frontend_filters(args, model, epoch):
    try:
        filters = model.frontend.weight.data
    except:
        raise TypeError
    one_norms = torch.sum(torch.abs(filters), dim=(tuple(range(1, filters.dim()))))

    filters = filters * 1 / one_norms.view(-1, 1, 1, 1)
    for i in range(5):
        for j in range(5):

            plt.subplot(5, 5, 5 * i + j + 1)
            img_abs = model.frontend.weight.abs().max()
            plt.imshow(
                model.frontend.weight[5 * i + j, 0, :, :].detach().cpu().numpy(),
                cmap=cm,
                interpolation="nearest",
                vmin=-img_abs,
                vmax=img_abs,
            )

            ax = plt.gca()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    plt.tight_layout()

    if not os.path.isdir("figs"):
        os.mkdir("figs")
    plt.savefig("./figs/" + str(args.dataset) + "_filters_" + str(epoch) + ".pdf")
    plt.close()

    filters = model.frontend.weight.data.detach().cpu().numpy()
    from numpy import savez

    if not os.path.isdir("filter_save"):
        os.mkdir("filter_save")
    savez("./filter_save/" + str(args.dataset) + "_" + str(epoch), filters=filters)


def plot_frontend_histogram(args, model, train_loader, epoch):
    """
    
    Plots the histogram of normalized activations

    """

    device = model.parameters().__next__().device
    nb_bins = 100
    cumulative_hist = torch.zeros(nb_bins).cpu()

    try:
        model.frontend.no_activation = True
    except:
        raise TypeError

    with torch.no_grad():
        for _, (images, _) in enumerate(train_loader):
            images = images.to(device)
            # gpu implementation doesn't work in pytorch1.4.0
            cumulative_hist.add_(
                torch.histc(model.frontend(images).cpu(), bins=nb_bins, min=-1, max=1)
            )

    model.frontend.no_activation = False

    fig = plt.figure(dpi=500, figsize=(6, 4))
    ax = plt.gca()
    cumulative_hist = cumulative_hist.detach().cpu().numpy()
    plt.bar(
        torch.linspace(-1, 1 - 2 / nb_bins, nb_bins),
        cumulative_hist,
        width=2 / nb_bins,
        align="edge",
    )
    plt.xlabel(r"frontend output $/||w_i||_1$", fontsize=15)
    plt.title("Histogram of normalized activations", fontsize=18)
    plt.ylim([0, min(cumulative_hist.max(), 0.3e9)])
    plt.tight_layout()

    if not os.path.isdir("figs"):
        os.mkdir("figs")
    plt.savefig("./figs/" + str(args.dataset) + "_hist_" + str(epoch) + ".pdf")
    plt.close()
    from numpy import savez

    if not os.path.isdir("hist_save"):
        os.mkdir("hist_save")
    savez("./hist_save/" + str(args.dataset) + "_" + str(epoch), hist=cumulative_hist)


def train(args, model, train_loader, optimizer, epoch):

    if args.savefig_train and (epoch % 5 == 0):
        plot_frontend_filters(args, model, epoch)
        plot_frontend_histogram(args, model, train_loader, epoch)

    device = model.parameters().__next__().device
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(images)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(preds, labels)

        if not isinstance(
            model.frontend, BuiltinFunctionType
        ):  # True means direct_quantization_model

            if epoch < args.stage2_start:
                model.frontend.no_activation = True
                loss += (
                    args.bs1
                    * epoch
                    / args.stage2_start
                    * torch.exp(
                        -((model.frontend(images)) ** 2) / (2 * args.bw1 ** 2)
                    ).mean()
                )

            elif epoch >= args.stage2_start and epoch < args.stage3_start:

                model.frontend.no_activation = True
                loss += (
                    (epoch - args.stage2_start)
                    / (args.stage3_start - args.stage2_start)
                    * args.bs2
                    * (
                        torch.exp(
                            -((model.frontend(images) - args.jump) ** 2)
                            / (2 * args.bw2 ** 2)
                        )
                        + torch.exp(
                            -((model.frontend(images) + args.jump) ** 2)
                            / (2 * args.bw2 ** 2)
                        )
                    ).mean()
                )

            elif epoch >= args.stage3_start:
                model.frontend.weight.requires_grad = False
                model.frontend.no_activation = False

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * args.batch_size
        _, pred_idx = torch.max(preds.data, 1)

        total += labels.size(0)
        correct += pred_idx.eq(labels.data).cpu().sum().float()

    return train_loss / 60000, 100.0 * correct / total


def test(args, model, test_loader, epoch):

    device = model.parameters().__next__().device
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    nb_batches = len(test_loader)

    if not isinstance(
        model.frontend, BuiltinFunctionType
    ):  # True means direct_quantization_model
        prev_activation_val = model.frontend.no_activation
        model.frontend.no_activation = False

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(images)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(preds, labels)

        test_loss += loss.item() * args.batch_size
        _, pred_idx = torch.max(preds.data, 1)
        total += labels.size(0)
        correct += pred_idx.eq(labels.data).cpu().sum().float()

    if not isinstance(
        model.frontend, BuiltinFunctionType
    ):  # True means direct_quantization_model
        model.frontend.no_activation = prev_activation_val

    return test_loss / 10000, 100.0 * correct / total


def save_checkpoint(args, model, acc, epoch):
    # print('=====> Saving checkpoint...')

    state = {
        "model_state_dict": model.state_dict(),
        "acc": acc,
        "epoch": epoch,
        "rng_state": torch.get_rng_state(),
    }
    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")
    torch.save(
        state, args.save_dir + args.checkpoint_name + "_epoch" + str(epoch) + ".ckpt"
    )


def adjust_lr(optimizer, epoch, args):
    lr = args.lr

    if epoch >= 50:
        lr /= 10
    if epoch >= 100:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders(args):

    mean, std = get_mean_std(args)
    if args.dataset == "mnist":

        transform_train = transforms.Compose(
            [
                # transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.data_dir, train=True, download=False, transform=transform_train
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.data_dir, train=False, download=False, transform=transform_test
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    elif args.dataset == "fashion":

        transform_train = transforms.Compose(
            [
                # transforms.RandomCrop(28, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                args.data_dir, train=True, download=False, transform=transform_train
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                args.data_dir, train=False, download=False, transform=transform_test
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    return train_loader, test_loader


def get_mean_std(args):

    if args.dataset == "mnist":
        in_channels = 1
        # mean=(0.1307,)
        # std=(0.3081,)
        mean = (0,)
        std = (1,)

    elif args.dataset == "fashion":
        in_channels = 1
        # mean=(0.28589922189712524,)
        # std=(0.3530242443084717,)
        mean = (0,)
        std = (1,)

    return (mean, std)


def main():

    args = get_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    START_EPOCH = 0

    batch_size = args.batch_size
    train_loader, test_loader = get_loaders(args)
    mean, std = get_mean_std(args)

    jump = (args.jump - mean[0]) / std[0]

    print("=====> Building model...")
    model = model_dict[args.model](
        in_channels=1, jump=jump, bpda_steepness=args.bpda_steepness,
    )

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("=====> Use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    if args.resume:
        print("=====> Resuming from checkpoint...")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch.load(args.save_dir + args.checkpoint_name + ".ckpt")
        state_dict = checkpoint["model_state_dict"]

        if "module" not in list(state_dict.keys())[0] and torch.cuda.device_count() > 1:
            # saved in single gpu machine, loading on multi gpu
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = "module." + key
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        elif "module" in list(state_dict.keys())[0] and not (
            torch.cuda.device_count() > 1
        ):
            # saved in multi gpu machine, loading on single gpu
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key[7:]
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        acc = checkpoint["acc"]
        START_EPOCH = checkpoint["epoch"]
        rng_state = checkpoint["rng_state"]
        torch.set_rng_state(rng_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not os.path.isdir("logs"):
        os.mkdir("logs")
    logname = "logs/" + args.checkpoint_name + ".csv"

    if not os.path.exists(logname):
        with open(logname, "w") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(
                ["Epoch", "Train Loss", "Train Acc", "Test Loss", "Test Acc"]
            )

    with tqdm(
        total=args.epochs,
        initial=START_EPOCH,
        unit="ep",
        unit_scale=True,
        unit_divisor=1000,
        leave=False,
        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
    ) as pbar:
        for epoch in range(START_EPOCH, args.epochs):

            train_loss, train_acc = train(args, model, train_loader, optimizer, epoch)
            test_loss, test_acc = test(args, model, test_loader, epoch)
            adjust_lr(optimizer, epoch, args)

            with open(logname, "a") as logfile:
                logwriter = csv.writer(logfile, delimiter=",")
                logwriter.writerow(
                    [
                        f"{epoch}, ",
                        f"{train_loss:.2f}, ",
                        f"{train_acc.item():.2f}, ",
                        f"{test_loss:.2f}, ",
                        f"{test_acc.item():.2f}",
                    ]
                )

            if (epoch + 1) % args.num_ckpt_steps == 0:
                save_checkpoint(args, model, test_acc, epoch)

            pbar.set_postfix(
                Loss=f"{test_loss:.2f}", Acc=f"{test_acc:.2f} %", refresh=True,
            )
            pbar.update(1)


if __name__ == "__main__":
    main()
