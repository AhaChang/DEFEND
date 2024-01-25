import os
import argparse
import tqdm
import gc
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch_geometric.seed import seed_everything
from torch_geometric.nn import MLP, GCN
from torch_geometric.utils import to_dense_adj

from model import DEFEND
from utils import sensitive_tensor_to_idx_dict, eval_scores


def predict(pred_score,threshold):
    prediction = (pred_score > threshold).astype(int).ravel()
    return prediction


def train_ae(args, model, data, bestmodel_path):
    patient = 20
    cur_p = 0
    best_loss = 9999
    for e in range(args.epoch0):
        model.train()
        cost_dict = model(data.x, data.edge_index, data.sensitive, mode='dvae_train')
        if args.verbose == 1:
            print('Epoch',e,'dvae_loss:',cost_dict['dvae_cost'].cpu().detach().numpy(), 'clf_loss:', cost_dict['clf_cost'].cpu().detach().numpy())
        if cost_dict['dvae_cost'].cpu().detach().numpy() < best_loss:
            best_loss = cost_dict['dvae_cost'].cpu().detach().numpy()
            cur_p = 0
            torch.save(model, bestmodel_path)
        else:
            cur_p = cur_p + 1

        if cur_p > patient:
            break
    
    return torch.load(bestmodel_path)

def train_ad(args, model, data, sensitive_dict, contamination):
    for e in range(args.epoch):
        model.train()
        score, cost_dict = model(data.x, data.edge_index, data.sensitive, mode='ad_train')
        prob = score
        threshold = np.percentile(prob.detach().cpu().numpy(), 100*(1-contamination))
        pred = (prob > threshold).long()
        auc_score, pr_score, sp_score, eo_score = eval_scores(prob.detach().cpu().numpy(), 
                                                              pred.detach().cpu().numpy(), 
                                                              data.y.cpu().numpy(),
                                                              sensitive_dict)
        if args.verbose == 1:
            print(e, cost_dict['main_cost'].item(), auc_score, sp_score, eo_score, pr_score)

    return auc_score, sp_score, eo_score, pr_score
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='reddit', help='reddit/twitter')
    parser.add_argument('--epoch0', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr0', type=float, default=0.001, help='0.001 for Reddit and 0.005 for twitter')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num_trials', default=5, type=int, help='Number of times to repeat experiment')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--verbose', default=1, type=int)

    parser.add_argument('--weight_corr', type=float, default=1e-10)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1.5)

    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    if args.data == "reddit":
        data = torch.load(f"data/reddit.pt")
    elif args.data == "twitter":
        data = torch.load(f"data/twitter.pt")
        data.contamination = data.contamination.item()
    
    data.x = data.x.float()
    data.y = data.y.bool()
    data.sensitive = data.sensitive.float()
    sensitive_dict = sensitive_tensor_to_idx_dict(data.sensitive)
    contamination = data.contamination
    data = data.to(device)

    aucs, prs, sps, eoos = [], [], [], []
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    param_str = args.data + '_' + str(args.alpha) + '_' + str(args.gamma) + '_' + str(args.weight_corr) + '_' + str(args.gpu)

    data.s = to_dense_adj(data.edge_index)[0]
    weight = torch.std(data.s).detach() / \
                    (torch.std(data.x).detach() + torch.std(data.s).detach())
    
    for i in tqdm.tqdm(range(args.num_trials)):
        bestmodel_path = os.path.join(os.getcwd(), 'best_models', current_time+param_str, 'vgae_' + str(i) + '.pth')
        directory = os.path.dirname(bestmodel_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model = DEFEND(in_dim=data.x.shape[1], 
                       lr=args.lr0,
                       weight_stru=1-weight,
                       weight_corr=args.weight_corr,
                       gamma=args.gamma,
                       alpha=args.alpha,
                       backbone_dec=MLP,
                       device=device)
        
        # first stage
        model = train_ae(args, model, data, bestmodel_path)

        # second stage
        auc_score, sp_score, eoo_score, pr_score = train_ad(args, model, data, sensitive_dict, contamination)

        aucs.append(auc_score)
        prs.append(pr_score)
        sps.append(sp_score)
        eoos.append(eoo_score)

        gc.collect()
        with torch.cuda.device(args.gpu):
            torch.cuda.empty_cache()
        del model

    # save results
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)
    pr_mean, pr_std = np.mean(prs), np.std(prs)
    sp_mean, sp_std = np.mean(sps), np.std(sps)
    eoo_mean, eoo_std = np.mean(eoos), np.std(eoos)
    
    print(f"AUC: {auc_mean:.4f}±{auc_std:.4f}|" + f"PR: {pr_mean:.4f}±{pr_std:.4f}|" + 
          f"SP: {sp_mean:.4f}±{sp_std:.4f}|" + f"EOO: {eoo_mean:.4f}±{eoo_std:.4f}|")

