
import torch
from torch import nn
import torch.nn.functional as F

class Saturation_activation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, jump, bpda_steepness):
       
        result = 0.5 * (torch.sign(x - jump) +
                        torch.sign(x + jump))
        
        return result


    # @staticmethod
    # def backward(ctx, grad_output):
    #     """
    #     Use this if you want an approximation of the saturation 
    #     activation funciton in the backward pass. Uses the derivative
    #     of 0.5*(tanh(bpda_steepness*(x-jump))+tanh(bpda_steepness*(x+jump)))
    #     """
    #     x, jump, bpda_steepness = ctx.saved_tensors
    #     grad_input = None

    #     def sech(x):
    #         return 1/torch.cosh(x)

    #     del_out_over_del_in = 0.5 * (bpda_steepness * sech(bpda_steepness*(x - jump))**2 + 
    #         bpda_steepness * sech(bpda_steepness * (x + jump))**2)
        
    #     grad_input = del_out_over_del_in * grad_output

    #     return grad_input, None, None

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class BPDA_whole_frontend(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, jump, weights, no_activation):
        one_norms = torch.sum(torch.abs(weights), dim=(
            tuple(range(1, weights.dim()))))

        # to avoid division by zero        
        one_norms.add_(1e-12)
        
        # divide activations by L1 norms
        x = x / one_norms.view(1,-1,1,1)
 
        if not no_activation:
            result = 0.5 * (torch.sign(x - jump) +
                            torch.sign(x + jump))

        # rescale it back by the L1 norms
        # x = x * one_norms

        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class Normalized_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, jump, bpda_steepness, 
        bias=False, no_activation=False, **kwargs):
        super(Normalized_Conv2d, self).__init__(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.weight.requires_grad = False
        self.set_bpda_steepness(bpda_steepness)
        self.set_jump(jump)
        self.no_activation=no_activation

    def set_bpda_steepness(self, bpda_steepness):
        if isinstance(bpda_steepness,torch.Tensor):
            self.bpda_steepness = nn.Parameter(bpda_steepness.float())
        else:
            self.bpda_steepness = nn.Parameter(torch.tensor(bpda_steepness, dtype=torch.float))        
        self.bpda_steepness.requires_grad=False

    def set_jump(self, jump):
        if isinstance(jump,torch.Tensor):
            self.jump = nn.Parameter(jump.float())
        else:
            self.jump = nn.Parameter(torch.tensor(jump, dtype=torch.float))
        self.jump.requires_grad=False
        
    def forward(self, x):
        x = super(Normalized_Conv2d, self).forward(x)
        one_norms = torch.sum(torch.abs(self.weight), dim=(
            tuple(range(1, self.weight.dim()))))

        # to avoid division by zero        
        one_norms.add_(1e-12)
        
        # divide activations by L1 norms
        x = x / one_norms.view(1,-1,1,1)
 
        if not self.no_activation:
            x = Saturation_activation().apply(x, self.jump, self.bpda_steepness)

        # rescale it back by the L1 norms
        # x = x * one_norms

        return x
    
    # def forward(self,x):
    #     """
    #     This uses identity for the whole frontend in the backward pass
    #     """
    #     x = super(Normalized_Conv2d, self).forward(x)
    #     return BPDA_whole_frontend().apply(x,self.jump, self.weight, self.no_activation)


