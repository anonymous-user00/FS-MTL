import random
import numpy as np
import copy
from typing import List, Tuple
from scipy.optimize import minimize
import torch


def PCGrad(grads: List[Tuple[torch.Tensor]], reduction: str = "sum") -> torch.Tensor:
    pc_grad = copy.deepcopy(grads)
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = sum(
                [
                    torch.dot(torch.flatten(grad_i), torch.flatten(grad_j))
                    for grad_i, grad_j in zip(g_i, g_j)
                ]
            )
            if g_i_g_j < 0:
                g_j_norm_square = (
                    torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
                )
                for grad_i, grad_j in zip(g_i, g_j):
                    grad_i -= g_i_g_j * grad_j / g_j_norm_square

    merged_grad = [sum(g) for g in zip(*pc_grad)]
    if reduction == "mean":
        merged_grad = [g / len(grads) for g in merged_grad]

    return merged_grad


def CAGrad(grads, alpha=0.5, rescale=1):
    n_tasks = len(grads)
    grads = grads.t()

    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    # GG = torch.zeros((n_tasks, n_tasks))
    # for i in range(n_tasks):
    #     for j in range(n_tasks):
    #         GG[i][j] = torch.dot(grads[i], grads[j]).cpu()
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(n_tasks) / n_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, n_tasks).dot(A).dot(b.reshape(n_tasks, 1))
            + c
            * np.sqrt(x.reshape(1, n_tasks).dot(A).dot(x.reshape(n_tasks, 1)) + 1e-8)
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)

def EW(grads_list):
    return sum(grads_list)

def IMTL(grads_list):
    grads = {}
    norm_grads = {}

    for i, grad in enumerate(grads_list):

        norm_term = torch.norm(grad)

        grads[i] = grad
        norm_grads[i] = grad / norm_term

    G = torch.stack(tuple(v for v in grads.values()))
    D = (
        G[
            0,
        ]
        - G[
            1:,
        ]
    )

    U = torch.stack(tuple(v for v in norm_grads.values()))
    U = (
        U[
            0,
        ]
        - U[
            1:,
        ]
    )
    first_element = torch.matmul(
        G[
            0,
        ],
        U.t(),
    )
    try:
        second_element = torch.inverse(torch.matmul(D, U.t()))
    except:
        # workaround for cases where matrix is singular
        second_element = torch.inverse(
            torch.eye(len(grads_list) - 1, device=norm_term.device) * 1e-8
            + torch.matmul(D, U.t())
        )

    alpha_ = torch.matmul(first_element, second_element)
    alpha = torch.cat(
        (torch.tensor(1 - alpha_.sum(), device=norm_term.device).unsqueeze(-1), alpha_)
    )
    return sum([alpha[i] * grads[i] for i in range(len(grads_list))])



def RLW(grads_list: List[torch.Tensor]) -> torch.Tensor:
    """
    RLW: sample a random simplex via softmax(randn); weight grads.
    """
    n = len(grads_list)
    w = torch.softmax(torch.randn(n, device=grads_list[0].device), dim=0)
    return sum(w[i] * grads_list[i] for i in range(n))

def SI(
    losses: List[torch.Tensor],
    grads_list: List[torch.Tensor],
    weights = None,
) -> torch.Tensor:
    """
    L = sum_i w_i * log(loss_i)
    ⇒ ∇ = sum_i w_i * (1/loss_i) * g_i
    losses and grads_list must align in order.
    """
    n = len(losses)
    if weights is None:
        weights = torch.ones(n, device=losses[0].device) / n
    agg = 0.0
    for w, l, g in zip(weights, losses, grads_list):
        agg = agg + w * g / (l.detach() + 1e-12)
    return agg
logsigma = None

def UW(
    grads_list: List[torch.Tensor],
    losses: List[torch.Tensor],
    logsigma: torch.Tensor,
) -> torch.Tensor:
    """
    Impart “uncertainty” weights:
      L = sum_i 0.5*(exp(-logsigma_i)*loss_i + logsigma_i)
    ⇒ ∇_θ L = sum_i 0.5*exp(-logsigma_i)*∇_θ loss_i

    :param grads_list: list of per-task gradients (same shape)
    :param losses:     list of per-task losses
    :param logsigma:   1D tensor of size n_tasks (learnable)
    :return:           aggregated gradient tensor
    """
    # weight each grad by 0.5 * exp(-logsigma_i)
    weights = 0.5 * torch.exp(-logsigma)
    agg = sum(w * g for w, g in zip(weights, grads_list))
    return agg


def DWA(
    grads_list: List[torch.Tensor],
    losses: List[torch.Tensor],
    costs: np.ndarray,
    iteration: int,
    iteration_window: int = 25,
    temp: float = 2.0,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Dynamic Weight Averaging:
      maintain a FIFO buffer `costs` of shape (2*window, n_tasks),
      on each call push current losses, pop oldest;
      after window steps compute:
        ws = mean(latest window costs) / mean(previous window costs)
        w = softmax(ws/temp)
      and use those as weights over grads.

    :param grads_list:      list of per-task gradients
    :param losses:          list of per-task losses (torch.Tensor scalars)
    :param costs:           np.ndarray[2*window, n_tasks] of past loss values
    :param iteration:       current iteration count (0-based)
    :param iteration_window: how many iters form each half of the buffer
    :param temp:            temperature
    :returns:               (aggregated gradient, updated costs buffer)
    """
    n_tasks = len(grads_list)
    # 1) append newest losses to costs buffer (drop oldest)
    new_row = np.array([l.detach().cpu().item() for l in losses], dtype=np.float32)
    costs = np.concatenate([costs[1:], new_row[None, :]], axis=0)

    # 2) compute DWA weights once we've filled one window
    if iteration >= iteration_window:
        first_half = costs[:iteration_window].mean(axis=0)
        second_half = costs[iteration_window:].mean(axis=0)
        ws = second_half / (first_half + 1e-12)
        raw = np.exp(ws / temp)
        w_norm = raw / raw.sum()
    else:
        # until then, just uniform
        w_norm = np.ones(n_tasks, dtype=np.float32) / n_tasks

    # 3) aggregate grads
    agg = sum(w_norm[i] * grads_list[i] for i in range(n_tasks))

    return agg, costs