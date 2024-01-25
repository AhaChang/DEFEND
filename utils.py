import numpy as np
import warnings
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

'''
metrics: statistical_parity, equality_of_odds
'''
def statistical_parity(pred, sensitive_var_dict):
    """
    Statistical parity of model prediction across different sensitive attribute values

    Parameters
    ----------
    pred : numpy.ndarray
        BINARY Outlier predictions (0 or 1) in shape of ``(N, )``where 1 represents outliers,
        0 represents normal instances.

    sensitive_var_dict: dictionary of key -> numpy.ndarray
        For each value of the sensitive attribute, a list of indexes that correspond to that value. 
        E.g. A list of indices for each gender in the dataset.

    Returns
    -------
    SP : float
        Statistical Parity score (0 to 1), calculated as the maximum rate of prediction y_hat minus minimum
        rate of prediction across all values v of the sensitive attribute X. Lower is better.  
        $\Delta_{SP} = \max_{v\in X}(P(\hat{y}=1 | X=v)) - \min_{v\in X}(P(\hat{y}=1 | X=v))$.
    """
    rates = []
    for v in sensitive_var_dict:
        rates.append(np.mean(pred[sensitive_var_dict[v]]))
    return max(rates) - min(rates)

def equality_of_odds(pred, true, sensitive_var_dict):
    """
    Equality of odds of model prediction across different sensitive attribute values for nodes that are outliers

    Parameters
    ----------
    pred : numpy.ndarray
        BINARY Outlier predictions (0 or 1) in shape of ``(N, )``where 1 represents outliers,
        0 represents normal instances.

    true : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers, 0 represents normal instances.

    sensitive_var_dict: dictionary of key -> numpy.ndarray
        For each value of the sensitive attribute, a list of indexes that correspond to that value. 
        E.g. A list of indices for each gender in the dataset.

    Returns
    -------
    EO : float
        Equality of odds score (0 to 1), calculated as the maximum rate of prediction y_hat minus minimum
        rate of prediction across all values v of the sensitive attribute X that are labelled as outliers.
        Lower is better.  
        $\Delta_{EO} = \max_{v\in X}(P(\hat{y}=1 | X=v, y=1)) - \min_{v\in X}(P(\hat{y}=1 | X=v, y=1))$.
    """
    sens_var = np.zeros_like(pred)
    number_of_values = len(sensitive_var_dict)
    for v in range(number_of_values):
        sens_var[sensitive_var_dict[v]] = v

    full = np.column_stack((pred, true, sens_var))
    only_true = full[full[:, 1] == 1]

    rates = []
    for v in range(number_of_values):
        temp = only_true[only_true[:, 2] == v]
        rates.append(np.mean(temp[:, 0]))

    if len(rates) < 2:
        warnings.warn("Positive labels only exist in <2 sensitive variable categories. EO value meaningless")

    return max(rates) - min(rates)


def sensitive_tensor_to_idx_dict(tensor):
    values, inverse_indices = torch.unique(tensor, return_inverse=True)
    values = values.tolist()
    idx_dict = dict()

    for i, v in enumerate(values):
        idx_dict[v] = (inverse_indices == i).nonzero()
    
    return idx_dict


def eval_scores(prob, pred, labels, sensitive_dict):
    sp_score = statistical_parity(pred, sensitive_dict)
    eo_score = equality_of_odds(pred, labels, sensitive_dict)
    auc_score = roc_auc_score(labels, prob)
    pr_score = average_precision_score(labels, prob)

    return auc_score, pr_score, sp_score, eo_score

def z_sampling_(mean, logvar, device):
    eps = torch.randn(mean.size()[0], mean.size()[1], device=device)
    return eps * torch.exp(logvar / 2) + mean
