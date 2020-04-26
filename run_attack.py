"""
Author: Can Bakiskan
Date: 2019-06-07

Attacks model with specified attack method, 
you can also attack blackbox using another model.

"""

import os
from tqdm import tqdm
import plot_settings
from plot_settings import cm
import matplotlib.pyplot as plt
from parameters import get_arguments

import numpy as np
import torch

from attacks import FastGradientSignMethod as FGSM
from attacks import ProjectedGradientDescent as PGD

# from attacks_alternative import FastGradientSignMethod as FGSM
# from attacks_alternative import ProjectedGradientDescent as PGD

from train import model_dict, get_loaders, get_mean_std
from models import *


def plot_image_samples(
    args,
    normalized_min,
    normalized_max,
    original_images_cpu,
    original_preds_cpu,
    perturbed_images_cpu,
    perturbed_preds_cpu,
    labels_cpu,
):

    plt.figure(dpi=200, figsize=(20, 5))

    correct_before = np.equal(original_preds_cpu, labels_cpu).astype(np.uint8)
    correct_after = np.equal(perturbed_preds_cpu, labels_cpu).astype(np.uint8)

    correct_both = correct_before * correct_after
    wrong_after_per = (correct_before - correct_after) == 1

    i = 0

    nb_rows = 3
    nb_cols = 13

    for j in range(nb_cols):
        if j == nb_cols // 2:
            continue
        elif j < nb_cols // 2:
            index = np.random.choice(np.nonzero(correct_both)[0])
        else:
            index = np.random.choice(np.nonzero(wrong_after_per)[0])

        plt.subplot(nb_rows, nb_cols, nb_cols * i + j + 1)
        plt.imshow(
            original_images_cpu[index, 0, :, :],
            vmin=normalized_min,
            vmax=normalized_max,
        )
        plt.xticks([])
        plt.yticks([])

        plt.subplot(nb_rows, nb_cols, nb_cols * (i + 1) + j + 1)

        plt.imshow(
            perturbed_images_cpu[index, 0, :, :],
            vmin=normalized_min,
            vmax=normalized_max,
        )
        plt.xticks([])
        plt.yticks([])

        plt.subplot(nb_rows, nb_cols, nb_cols * (i + 2) + j + 1)

        difference_img = (perturbed_images_cpu - original_images_cpu)[index, 0, :, :]

        img_min = difference_img.min()
        img_max = difference_img.max()
        img_abs = np.abs(difference_img).max()

        if img_abs < 0.01:
            img_abs = 1

        plt.imshow(
            difference_img,
            cmap=cm,
            interpolation="nearest",
            vmin=-img_abs,
            vmax=img_abs,
        )
        plt.xticks([])
        plt.yticks([])
        del difference_img

    plt.tight_layout(0, 0)

    current_path = os.path.dirname(os.path.realpath(__file__))
    filename = args.checkpoint_name + "_attack.pdf"
    path = os.path.join(current_path, "figs", filename)
    plt.savefig(path)
    plt.close()


def main():

    args = get_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    START_EPOCH = 0

    batch_size = args.batch_size
    train_loader, test_loader = get_loaders(args)
    mean, std = get_mean_std(args)

    normalized_min = (0 - mean[0]) / std[0]
    normalized_max = (1 - mean[0]) / std[0]
    epsilon_normalized = args.eps / std[0]
    step_size_normalized = args.step_size / std[0]
    jump = (args.jump - mean[0]) / std[0]

    print("=====> Loading checkpoint...")

    model = model_dict[args.model](
        in_channels=1, jump=jump, bpda_steepness=args.bpda_steepness,
    )
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("=====> Use", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    assert os.path.isdir("checkpoints"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoints/" + args.checkpoint_name + ".ckpt")
    state_dict = checkpoint["model_state_dict"]

    if "module" not in list(state_dict.keys())[0] and torch.cuda.device_count() > 1:
        # saved in single gpu machine, loading on multi gpu
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = "module." + key
            new_state_dict[new_key] = state_dict[key]
        state_dict = new_state_dict

    elif "module" in list(state_dict.keys())[0] and not (torch.cuda.device_count() > 1):
        # saved in multi gpu machine, loading on single gpu
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key[7:]
            new_state_dict[new_key] = state_dict[key]
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

    # turn train mode off for batchnorm, dropout etc
    model.eval()

    print("=====> Done.")

    original_images_cpu = np.zeros((10000, 1, 28, 28))
    perturbed_images_cpu = np.zeros((10000, 1, 28, 28))
    original_preds_cpu = np.zeros(10000)
    perturbed_preds_cpu = np.zeros(10000)
    labels_cpu = np.zeros(10000)

    cpu_array_index = 0

    for images, labels in tqdm(test_loader):

        if args.attack_method == "fgsm":
            images = images.to(device)
            labels = labels.to(device)
            perturbation = FGSM(
                model,
                images,
                labels,
                epsilon_normalized,
                data_params={"x_min": normalized_min, "x_max": normalized_max,},
            )

            perturbed_images = torch.clamp(
                images + perturbation, normalized_min, normalized_max
            )

        elif args.attack_method == "r_iter":
            images = images.to(device)
            labels = labels.to(device)

            perturbation = PGD(
                model,
                images,
                labels,
                show_bar=True,
                data_params={"x_min": normalized_min, "x_max": normalized_max,},
                attack_params={
                    "norm": "inf",
                    "eps": epsilon_normalized,
                    "step_size": step_size_normalized,
                    "num_steps": args.num_steps,
                    "random_start": True,
                    "num_restarts": 1,
                },
            )

            perturbed_images = torch.clamp(
                images + perturbation, normalized_min, normalized_max
            )

        elif args.attack_method == "pgd":
            images = images.to(device)
            labels = labels.to(device)

            perturbation = PGD(
                model,
                images,
                labels,
                show_bar=True,
                data_params={"x_min": normalized_min, "x_max": normalized_max,},
                attack_params={
                    "norm": "inf",
                    "eps": epsilon_normalized,
                    "step_size": step_size_normalized,
                    "num_steps": args.num_steps,
                    "random_start": True,
                    "num_restarts": args.num_restarts,
                },
            )

            perturbed_images = torch.clamp(
                images + perturbation, normalized_min, normalized_max
            )

        original_images = images
        perturbed_preds = torch.argmax(model(perturbed_images), dim=1)
        original_preds = torch.argmax(model(original_images), dim=1)

        perturbed_images_cpu[cpu_array_index : cpu_array_index + batch_size] = (
            perturbed_images.detach().cpu().numpy()
        )
        original_images_cpu[cpu_array_index : cpu_array_index + batch_size] = (
            original_images.detach().cpu().numpy()
        )

        perturbed_preds_cpu[cpu_array_index : cpu_array_index + batch_size] = (
            perturbed_preds.detach().cpu().numpy()
        )
        original_preds_cpu[cpu_array_index : cpu_array_index + batch_size] = (
            original_preds.detach().cpu().numpy()
        )
        labels_cpu[cpu_array_index : cpu_array_index + batch_size] = (
            labels.detach().cpu().numpy()
        )

        cpu_array_index += batch_size

        torch.cuda.empty_cache()

    perturbed_accuracy = (
        np.equal(perturbed_preds_cpu, labels_cpu).astype(np.float).mean()
    )

    original_accuracy = np.equal(original_preds_cpu, labels_cpu).astype(np.float).mean()

    print("original accuracy: %.2f %%" % (100 * original_accuracy))
    print("perturbed accuracy: %.2f %%" % (100 * perturbed_accuracy))

    if args.save_attack:
        import h5py

        current_path = os.path.dirname(os.path.realpath(__file__))
        filename = (
            args.checkpoint_name
            + "_attack_"
            + args.attack_method
            + "_eps"
            + str(args.eps)
            + ".h5"
        )
        path = os.path.join(current_path, "attack_save", filename)
        h5f = h5py.File(path, "w")

        h5f.create_dataset("perturbed_images", data=perturbed_images_cpu)
        h5f.create_dataset("perturbed_preds", data=perturbed_preds_cpu)
        h5f.create_dataset("labels", data=labels_cpu)
        h5f.create_dataset("original_images", data=original_images_cpu)
        h5f.create_dataset("original_preds", data=original_preds_cpu)
        h5f.create_dataset(
            "accuracies", data=np.array([original_accuracy, perturbed_accuracy])
        )
        h5f.close()

    if args.savefig_attack:
        plot_image_samples(
            args,
            normalized_min,
            normalized_max,
            original_images_cpu,
            original_preds_cpu,
            perturbed_images_cpu,
            perturbed_preds_cpu,
            labels_cpu,
        )


if __name__ == "__main__":
    main()
