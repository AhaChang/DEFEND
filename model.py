import torch
import torch.nn as nn
import torch.nn.functional as F

from VGAE import VGAEBase
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import MLP, GCN
from pygod.nn.decoder import DotProductDecoder
from utils import z_sampling_


class DEFEND(nn.Module):
    """
    Initializes DEFEND network: VGAE encoder, MLP discriminator, Detector
    hid_num: hidden dimension of the encoder/decoder
    num_layers: hidden layer number of the encoder/decoder
    hid_num_d: hidden dimension of the MLP discriminator
    num_layers_d: hidden layer number of the MLP discriminator
    alpha: weight of the predictiveness term(e.g., logp(a|b))
    gamma: weight of the disentanglement term(e.g., KL(q(z,b)||q(z))\prod_{j}q(b_j))
    weight_corr: weight of the correlation term in anomaly detection
    weight_stru: weight of the structure reconstruction term
    """
    def __init__(self, 
                 in_dim, 
                 hid_dim=64, 
                 num_layers=2,
                 hid_num_d=16,
                 num_layers_d=2,
                 lr=0.01,
                 alpha=1,
                 gamma=0.5,
                 weight_corr=1e-9,
                 weight_stru=0.2,
                 batch_size=0,
                 act=F.relu,
                 backbone_dec=MLP,
                 device=torch.device("cpu")):
        super(DEFEND, self).__init__()

        self.lr = lr
        self.weight_stru = weight_stru
        self.gamma = gamma
        self.alpha = alpha
        self.weight_corr = weight_corr
        self.batch_size = batch_size
        self.device = device
        
        self.backbone_dec = backbone_dec

        # VGAE
        self.vgae = VGAEBase(in_dim=in_dim, 
                             hid_dim=hid_dim, 
                             num_layers=num_layers,
                             device=device).to(device)

        # MLP discriminator: for disentangle loss
        self.discriminator = MLP(in_channels=hid_dim, 
                                 hidden_channels=hid_num_d, 
                                 num_layers=num_layers_d+1, 
                                 out_channels=2, 
                                 act=F.relu).to(device)

        self.x_decoder = self.backbone_dec(in_channels=hid_dim, 
                             hidden_channels=hid_dim,
                             num_layers=num_layers,
                             out_channels=in_dim,
                             act=F.relu).to(device)

        # index for sensitive attribute
        self.n_sens = 1
        self.sens_idx = list(range(self.n_sens))
        self.nonsens_idx = [
            i for i in range(int(hid_dim)) if i not in self.sens_idx
        ]

        self.optimizer_dvae = torch.optim.Adam(self.vae_params(), lr=self.lr)
        self.optimizer_disc = torch.optim.Adam(self.discriminator_params(), lr=self.lr)
        self.optimizer_dec = torch.optim.Adam(self.dec_params(),lr=self.lr)

    def vae_params(self):
        """Returns VAE parameters required for training VAE"""
        return list(self.vgae.parameters())

    def discriminator_params(self):
        """Returns discriminator parameters"""
        return list(self.discriminator.parameters())

    def dec_params(self):
        return list(self.x_decoder.parameters()) 


    def forward(self, x, edge_index, attrs, mode="train"):
        adj = to_dense_adj(edge_index)[0]
        # encode: get q(z,b|x)
        _mu, _logvar, _z = self.vgae.encoder(x, edge_index)

        # distribution of 'z' (non-sensitive)
        mu = _mu[:, self.nonsens_idx]
        logvar = _logvar[:, self.nonsens_idx]
        
        # the rest are 'b', deterministically modeled as logits of sens attrs a
        b_logits = _mu[:, self.sens_idx]

        # non-sensitive representation
        z = z_sampling_(mu, logvar, self.device) 

        # reparametrization
        zb = torch.zeros_like(_mu)
        zb[:, self.sens_idx] = b_logits
        zb[:, self.nonsens_idx] = z
        

        # optimization
        if mode == "dvae_train":
            # decode: get p(x|z,b)
            x_rec, adj_rec = self.vgae.decoder(zb, edge_index)

            # attribute reconstruction loss
            recon_term = F.mse_loss(x_rec, x, reduction='none').mean(1)

            # structure reconstruction loss
            edge_recon_loss = F.binary_cross_entropy(torch.sigmoid(adj_rec), adj, reduction='none').mean(1)

            std = (logvar / 2).exp()
            # get q(z|x)
            q_zIx = torch.distributions.Normal(mu, std)
            # prior: get p(z)
            p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            # compute analytic KL from q(z|x) to p(z)
            kl = torch.distributions.kl_divergence(q_zIx, p_z).sum(1)

            # VAE loss
            vae_loss =  (1 - self.weight_stru) * recon_term + self.weight_stru * edge_recon_loss + kl

            # predictiveness loss: get p(a|b)
            clf_losses = nn.BCEWithLogitsLoss()(b_logits.squeeze(),attrs)

            # disentanglement loss
            logits_joint = self.discriminator(zb)
            total_corr = logits_joint[:, 0] - logits_joint[:, 1]

            dvae_loss = vae_loss.mean() + self.gamma * total_corr.mean() + self.alpha * clf_losses

            # shuffling minibatch indexes of b0, b1, z
            z_fake = torch.zeros_like(zb)
            z_fake[:, 0] = zb[:, 0][torch.randperm(zb.shape[0])]
            z_fake[:, 1:] = zb[:, 1:][torch.randperm(zb.shape[0])]
            z_fake = z_fake.to(self.device).detach()

            # discriminator
            logits_joint_prime = self.discriminator(z_fake)
            if self.batch_size == 0:
                self.batch_size = x.shape[0]
            ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
            zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            disc_loss = (
                0.5
                * (
                    F.cross_entropy(logits_joint, zeros)
                    + F.cross_entropy(logits_joint_prime, ones)
                ).mean()
            )

            self.optimizer_dvae.zero_grad()
            dvae_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.vae_params(), 5.0)
            
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator_params(), 5.0)

            self.optimizer_dvae.step()
            self.optimizer_disc.step()

            # total cost
            cost_dict = dict(
                dvae_cost=dvae_loss, disc_cost=disc_loss, clf_cost=clf_losses
            )

            return cost_dict


        elif mode == "ad_train":
            # detector
            encoded_x = zb.clone().detach()
            
            # IMPORTANT: randomizing sensitive latent
            encoded_x[:, self.sens_idx] = torch.randn_like(encoded_x[:, self.sens_idx])

            if self.backbone_dec == GCN:
                non_sens_x_recon = self.x_decoder(encoded_x, edge_index)
            else:
                non_sens_x_recon = self.x_decoder(encoded_x)

            # attribute reconstruction loss
            non_sens_x_recon_score = F.mse_loss(non_sens_x_recon,x, reduction='none').mean(dim=1)
            non_sens_x_recon_loss = non_sens_x_recon_score.mean()
            
            recon_score = non_sens_x_recon_score
            
            # absolute correlation 
            recon_err_mean = torch.mean(recon_score)
            recon_err_std = torch.sqrt(torch.var(recon_score, unbiased=False))
            recon_err_centered = (torch.sum(recon_score) - recon_err_mean) / recon_err_std

            pred_b = b_logits.clone().detach()
            sens_var_mean = torch.mean(pred_b)
            sens_var_std = torch.sqrt(torch.var(pred_b, unbiased=False))
            sens_var_centered = (torch.sum(pred_b) - sens_var_mean) / sens_var_std

            corr_loss = torch.abs(recon_err_centered * sens_var_centered)

            loss_ad = self.weight_corr * corr_loss + non_sens_x_recon_loss

            self.optimizer_dec.zero_grad()
            loss_ad.backward()
            torch.nn.utils.clip_grad_norm_(self.dec_params(), 5.0)
            self.optimizer_dec.step()

            # total cost
            cost_dict = dict(
                main_cost=loss_ad
            )
    
            return recon_score, cost_dict

    def get_embeds(self, x, edge_index, non_s=True):
        # encode: get q(z,b|x)
        _mu, _logvar, _z = self.vgae.encoder(x, edge_index)

        if non_s:
            # distribution of 'z' (non-sensitive)
            mu = _mu[:, self.nonsens_idx]
            logvar = _logvar[:, self.nonsens_idx]
            return mu.detach().cpu().numpy()
        else: 
            # IMPORTANT: randomizing sensitive latent
            _mu[:, self.sens_idx] = torch.randn_like(_mu[:, self.sens_idx])
            return _mu.detach().cpu().numpy()