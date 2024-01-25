import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN
from torch_geometric.utils import to_dense_adj, negative_sampling
from pygod.detector import DeepDetector
from pygod.nn.decoder import DotProductDecoder
import math
from utils import z_sampling_


class VGAEBase(nn.Module):
    def __init__(self, 
                 in_dim,
                 hid_dim=64,
                 num_layers=2,
                 dropout=0.,
                 act=F.relu,
                 backbone=GCN,
                 sigmoid_s=False,
                 device=None,
                 **kwargs):
        super(VGAEBase, self).__init__()

        self.hid_dim = hid_dim
        self.device = device

        self.enc_layers = backbone(in_channels=in_dim, 
                                   hidden_channels=hid_dim, 
                                   num_layers=num_layers,
                                   out_channels=hid_dim*2, 
                                   act=act, 
                                   dropout=dropout,
                                   **kwargs)
        
        self.dec_layers = backbone(in_channels=hid_dim, 
                                   hidden_channels=hid_dim, 
                                   num_layers=num_layers,
                                   out_channels=in_dim, 
                                   act=act, 
                                   dropout=dropout,
                                   **kwargs)
        
        self.struc_dec_layers = DotProductDecoder(in_dim=hid_dim,
                                                  hid_dim=hid_dim,
                                                  num_layers=num_layers-1,
                                                  dropout=dropout,
                                                  act=act,
                                                  sigmoid_s=sigmoid_s,
                                                  backbone=backbone,
                                                  **kwargs)

    def encoder(self, x, edge_index):
        ouput = self.enc_layers(x, edge_index)
        z_mean, z_log_std = ouput[:,:self.hid_dim], ouput[:,self.hid_dim:]
        sampled_z = z_sampling_(z_mean, z_log_std, self.device)

        return z_mean, z_log_std, sampled_z

    def decoder(self, z, edge_index):
        x_ = self.dec_layers(z, edge_index)
        adj_ = self.struc_dec_layers(z, edge_index)

        return x_, adj_

    def forward(self, x, edge_index):
        z_mean, z_log_std, z = self.encoder(x, edge_index)
        x_rec, adj_rec = self.decoder(z, edge_index)

        return z_mean, z_log_std, x_rec, adj_rec
    
    @staticmethod
    def process_graph(data, recon_s=False):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        recon_s : bool, optional
            Reconstruct the structure instead of node feature .
        """
        if recon_s:
            data.s = to_dense_adj(data.edge_index)[0]


class GAEBase(nn.Module):
    def __init__(self, 
                 in_dim,
                 hid_dim=64,
                 num_layers=2,
                 dropout=0.,
                 act=F.relu,
                 backbone=GCN,
                 sigmoid_s=False,
                 device=None,
                 **kwargs):
        super(GAEBase, self).__init__()

        self.hid_dim = hid_dim
        self.device = device

        self.enc_layers = backbone(in_channels=in_dim, 
                                   hidden_channels=hid_dim, 
                                   num_layers=num_layers,
                                   out_channels=hid_dim, 
                                   act=act, 
                                   dropout=dropout,
                                   **kwargs)
        
        self.dec_layers = backbone(in_channels=hid_dim, 
                                   hidden_channels=hid_dim, 
                                   num_layers=num_layers,
                                   out_channels=in_dim, 
                                   act=act, 
                                   dropout=dropout,
                                   **kwargs)
        
        self.struc_dec_layers = DotProductDecoder(in_dim=hid_dim,
                                                  hid_dim=hid_dim,
                                                  num_layers=num_layers-1,
                                                  dropout=dropout,
                                                  act=act,
                                                  sigmoid_s=sigmoid_s,
                                                  backbone=backbone,
                                                  **kwargs)

    def encoder(self, x, edge_index):
        z = self.enc_layers(x, edge_index)

        return z

    def decoder(self, z, edge_index):
        x_ = self.dec_layers(z, edge_index)
        adj_ = self.struc_dec_layers(z, edge_index)

        return x_, adj_

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_rec, adj_rec = self.decoder(z, edge_index)

        return x_rec, adj_rec
    
    @staticmethod
    def process_graph(data, recon_s=False):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        recon_s : bool, optional
            Reconstruct the structure instead of node feature .
        """
        if recon_s:
            data.s = to_dense_adj(data.edge_index)[0]