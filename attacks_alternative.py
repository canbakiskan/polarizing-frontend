"""
Author: Can Bakiskan
Date: 2019-06-07

Attacks model with specified attack method, 
you can also attack blackbox using another model.

"""


from tqdm import tqdm
import torch
from torch import nn


def FastGradientSignMethod(model, x, y_true, epsilon, data_params):
    x.requires_grad = True
    y_hat = model(x)

    # keep track so we don't attack whats already wrong
    originally_right_indices = torch.argmax(y_hat, dim=1) == y_true
    # reduction none prohibits taking avg so we can take derivative of each image's loss independently
    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(y_hat, y_true)

    if loss.min() <= 0:

        y_true_onehot = torch.zeros_like(y_hat)
        y_true_onehot[torch.arange(y_hat.shape[0]), y_true] = 1.0
        loss[loss == 0.0] = nn.functional.mse_loss(
            y_hat[loss == 0.0], y_true_onehot[loss == 0.0], reduction="none"
        ).mean(dim=1)
        # raise GradientMaskingError("Gradient masking is happening")

    # gradient parameter specifies "vector" in jacobian vector product
    # it must be there so we can take gradient of multi variables (loss)
    loss.backward(
        gradient=torch.ones_like(y_true, dtype=torch.float), retain_graph=True
    )

    grad_wrt_img = x.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = grad_wrt_img.sign()
    perturbation = epsilon * sign_data_grad

    # mask the correctly labelled images
    # .view because pytorch requires number of dimensions to be the same
    # note that size of first dimension the same
    perturbation = originally_right_indices.view(-1, 1, 1, 1).float() * perturbation

    perturbation.data = torch.max(
        torch.min(perturbation, data_params["x_max"] - x), data_params["x_min"] - x,
    )
    # this returns perturbation, clamping to [0,1] done outside
    return perturbation


def ProjectedGradientDescent(
    model, images, y_true, show_bar, data_params, attack_params
):

    epsilon = attack_params["eps"]
    step_size = attack_params["step_size"]
    num_steps = attack_params["num_steps"]
    num_restarts = attack_params["num_restarts"]
    normalized_min = data_params["x_min"]
    normalized_max = data_params["x_max"]

    # keeps track across restarts
    max_dmg_perturbation = torch.zeros_like(images)

    originally_right_indices = torch.argmax(model(images), dim=1) == y_true

    # reduction none prohibits taking avg so we can take derivative of each image's loss independently
    criterion = nn.CrossEntropyLoss(reduction="none")
    # we want to never go below the original loss for each image, so start with original as max
    max_loss = criterion(model(images), y_true)

    for i in tqdm(range(num_restarts)):

        # random restart
        adv = 2 * epsilon * torch.rand_like(images) - epsilon * torch.ones_like(images)

        # turns off random restart
        # adv = torch.zeros_like(images)

        for i in tqdm(range(num_steps)):
            adv += FastGradientSignMethod(
                model,
                torch.clamp(images + adv, normalized_min, normalized_max),
                y_true,
                step_size,
                data_params,
            )
            adv = torch.clamp(adv, -epsilon, epsilon)

        output = model(torch.clamp(images + adv, normalized_min, normalized_max))

        loss = criterion(output, y_true)

        # if current loss is higher, restart successful
        update_indices = torch.le(max_loss, loss)

        # mask update indices
        update_indices = update_indices * originally_right_indices

        max_loss[update_indices] = loss[update_indices].data

        max_dmg_perturbation[update_indices] = adv[update_indices].data

        del loss
        del adv
        del output
        del update_indices
        torch.cuda.empty_cache()

    return max_dmg_perturbation
