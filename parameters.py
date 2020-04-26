"""
Hyper-parameters
"""

import argparse


def get_arguments():
    """ Hyper-parameters """

    parser = argparse.ArgumentParser(
        description="Train Polarization_quantization_model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets",
        metavar="datapath",
        help="Location of data files (datasets)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        metavar="N",
        help="number of epochs to train (default: 60)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )

    parser.add_argument("--decay", type=float, default=0, help="weight decay")

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints/",
        metavar="savepath",
        help="save path",
    )

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="polarized_model",
        metavar="name",
        help="name of checkpoint",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion",
        metavar="mnist/fashion",
        help="Which dataset to use (default: fashion)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Polarization_quantization_model",
        help="Which model model to train (default: Polarization_quantization_model)",
    )

    parser.add_argument(
        "--bpda_steepness",
        type=int,
        default=8,
        metavar="bs",
        help="backward pass differentiable approximation steepness (default: 8)",
    )

    parser.add_argument(
        "--jump",
        type=float,
        default=0.2,
        metavar="jump",
        help="jump of saturation activation function (default: 0.2)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="LR",
        help="batch size (default: 64)",
    )

    parser.add_argument(
        "--num_ckpt_steps",
        type=int,
        default=10,
        help="save checkpoint steps (default: 10)",
    )

    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )

    parser.add_argument(
        "--savefig_attack",
        action="store_true",
        help="plot correctly and wrongly classified images after attack",
    )

    parser.add_argument(
        "--savefig_train",
        action="store_true",
        help="plot filters and histograms in each epoch of training",
    )

    parser.add_argument(
        "--bs1", type=float, default=1, metavar="bump_scale1", help="bump scale stage 1"
    )

    parser.add_argument(
        "--bw1",
        type=float,
        default=0.35,
        metavar="bump_width1",
        help="bump width stage 1",
    )

    parser.add_argument(
        "--bs2", type=float, default=1, metavar="bump_scale2", help="bump scale stage 2"
    )

    parser.add_argument(
        "--bw2",
        type=float,
        default=0.15,
        metavar="bump_width2",
        help="bump width stage 2",
    )

    parser.add_argument(
        "--stage2_start",
        type=int,
        default=20,
        metavar="stage2_epoch",
        help="Start epoch of stage 2",
    )

    parser.add_argument(
        "--stage3_start",
        type=int,
        default=40,
        metavar="stage3_epoch",
        help="Start epoch of stage 3",
    )

    parser.add_argument(
        "--attack_method",
        type=str,
        default="pgd",
        metavar="fgsm/pgd/r_iter",
        help="Attack method to be used",
    )

    parser.add_argument(
        "--attack_batch_size",
        type=int,
        default=1000,
        metavar="LR",
        help="batch size (default: 64)",
    )

    parser.add_argument(
        "--save_attack",
        action="store_true",
        default=False,
        help="whether to save attack images",
    )

    parser.add_argument(
        "--eps", type=float, default=0.1, metavar="eps", help="Attack budget epsilon"
    )

    parser.add_argument(
        "--step_size",
        type=float,
        default=0.01,
        metavar="eps-itr",
        help="(Iterative attacks) Attack budget in each iteration",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        metavar="N",
        help="(Iterative attacks) Number of iterations in attack",
    )

    parser.add_argument(
        "--num_restarts",
        type=int,
        default=20,
        metavar="N",
        help="(PGD) Number of random restarts in attack",
    )

    args = parser.parse_args()

    return args
