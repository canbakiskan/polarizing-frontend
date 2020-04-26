"""
Author: Can Bakiskan
Date: 2019-08-07

"""

from tqdm import tqdm
import torch
from torch import nn


class GradientMaskingError(ValueError):
    """Gradient masking error"""

    def __init__(self, arg):
        self.arg = arg


def FastGradientSignMethod(model, x, y_true, eps, data_params):
    """
    Input :
        model : Neural Network (Classifier)
        x : Inputs to the model
        y_true : Labels
        eps : attack budget
        data_params : dictionary containing x_min and x_max
    Output:
        perturbation : Single step perturbation
    """
    e = torch.zeros_like(x, requires_grad=True)
    if x.device.type == "cuda":
        y_hat = model(x + e).type(torch.cuda.DoubleTensor)
    else:
        y_hat = model(x + e).type(torch.DoubleTensor)
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(y_hat, y_true)

    if loss.min() <= 0:

        y_true_onehot = torch.zeros_like(y_hat)
        y_true_onehot[torch.arange(y_hat.shape[0]), y_true] = 1.0
        loss[loss == 0.0] = nn.functional.mse_loss(
            y_hat[loss == 0.0], y_true_onehot[loss == 0.0], reduction="none"
        ).mean(dim=1)
        # raise GradientMaskingError("Gradient masking is happening")

    loss.backward(
        gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True
    )

    e_grad = e.grad.data
    perturbation = eps * e_grad.sign()

    perturbation.data = torch.max(
        torch.min(perturbation, data_params["x_max"] - x), data_params["x_min"] - x
    )
    return perturbation


def ProjectedGradientDescent(
    model,
    x,
    y_true,
    show_bar=True,
    data_params={"x_min": 0, "x_max": 1},
    attack_params={
        "norm": "inf",
        "eps": 8.0 / 255.0,
        "step_size": 8.0 / 255.0 / 10,
        "num_steps": 100,
        "random_start": True,
        "num_restarts": 1,
    },
):
    """
    Input :
        model : Neural Network (Classifier)
        x : Inputs to the model
        y_true : Labels
        show_bar: Display loading bar
        data_params: Data parameters as dictionary
                x_min : Minimum legal value for elements of x
                x_max : Maximum legal value for elements of x
        attack_params : Attack parameters as a dictionary
                norm : Norm of attack
                eps : Attack budget
                step_size : Attack budget for each iteration
                num_steps : Number of iterations
                random_start : Randomly initialize image with perturbation
                num_restarts : Number of restarts
    Output:
        perturbs : Perturbations for given batch
    """

    # fooled_indices = np.array(y_true.shape[0])
    perturbs = torch.zeros_like(x)

    if show_bar and attack_params["num_restarts"] > 1:
        restarts = tqdm(range(attack_params["num_restarts"]))
    else:
        restarts = range(attack_params["num_restarts"])

    for i in restarts:
        if attack_params["random_start"] or attack_params["num_restarts"] > 1:
            perturb = (2 * torch.rand_like(x) - 1) * attack_params["eps"]

        else:
            perturb = torch.zeros_like(x, dtype=torch.float)

        if show_bar:
            iters = tqdm(range(attack_params["num_steps"]))
        else:
            iters = range(attack_params["num_steps"])

        for _ in iters:
            perturb += FastGradientSignMethod(
                model,
                torch.clamp(x + perturb, data_params["x_min"], data_params["x_max"]),
                y_true,
                attack_params["step_size"],
                data_params,
            )
            perturb = torch.clamp(perturb, -attack_params["eps"], attack_params["eps"])

        if i == 0:
            perturbs = perturb.data
        else:
            output = model(
                torch.clamp(x + perturb, data_params["x_min"], data_params["x_max"])
            )
            y_hat = output.argmax(dim=1, keepdim=True)

            fooled_indices = (y_hat != y_true.view_as(y_hat)).nonzero()
            perturbs[fooled_indices] = perturb[fooled_indices].data

    perturbs.data = torch.max(
        torch.min(perturbs, data_params["x_max"] - x), data_params["x_min"] - x
    )

    return perturbs
