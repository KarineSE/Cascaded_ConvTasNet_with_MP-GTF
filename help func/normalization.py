import numpy as np
import torch


def normalize_estimates(est_np, mix_np):
    """Normalizes estimates according to the mixture maximum amplitude

    Args:
        est_np (np.array): Estimates with shape (n_src, time).
        mix_np (np.array): One mixture with shape (time, ).

    """
    mix_max = np.max(np.abs(mix_np))
    return np.stack([est * mix_max / np.max(np.abs(est)) for est in est_np])


def normalize_estimates_cascaded(est_np, mix_np):
    """Normalizes estimates according to WHMAR! article

    Args:
        est_np (np.array): Estimates with shape (n_src, time).
        mix_np (np.array): One mixture with shape (time, ).

    """
    return np.stack([est * np.dot(est, mix_np) / np.dot(est, est) for est in est_np])


def normalize_estimates_cascaded_tensor(est_tensor, mix_tensor):
    """Normalizes estimates according to WHMAR! article

    Args:
        est_np (np.array): Estimates with shape (n_src, time).
        mix_np (np.array): One mixture with shape (time, ).

    """
    # for i, x in enumerate(est):
    #     for j, est in enumerate(x):
    #         est * np.dot(est, mix) / np.dot(est, est)

    return torch.stack([torch.stack([est_tensor[i, j, :] * torch.dot(est_tensor[i, j, :], mix_tensor[i, :]) / torch.dot(est_tensor[i, j, :], est_tensor[i, j, :])
                                 for j in range(est_tensor.shape[1])]) for i in range(mix_tensor.shape[0])])

    # return np.stack([est * np.dot(est, mix) / np.dot(est, est) for est in est_tensor])
