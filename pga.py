
from typing import Optional
from math import sqrt

import os
import time
import torch
torch.set_printoptions(threshold=1000000)
import numpy as np
np.set_printoptions(threshold=10e6)
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx, to_dense_adj
from utils import k_hop_subgraph_with_default_whole_graph
from Configures import model_args, data_args, explainer_args

EPS = 1e-6


def inv_sigmoid(t: torch.Tensor):
    """ except the case t is 0 or 1 """
    if t.shape[0] != 0:
        if t.min().item() == 0 or t.max().item() == 1:
            t = 0.99 * t + 0.005
    ret = - torch.log(1 / t - 1)
    return ret

def gather_nd(params, indices):
    '''
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices
    
    returns: tensor shaped [m_1, m_2, m_3, m_4]
    
    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices
    
    returns: tensor shaped [m_1, ..., m_1]
    '''

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1) # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1
    
    for i in range(ndim)[::-1]:
        idx += indices[i] * m 
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)

class PGA(nn.Module):

    coeffs = {
        'node_feat_size': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, F, epochs: int = 20, lr: float = 0.003,
                 top_k: int = 20, num_hops: Optional[int] = None):
        # lr=0.005, 0.003
        super(PGA, self).__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.top_k = top_k
        self.__num_hops__ = num_hops
        self.device = model.device

        self.coff_ratio = explainer_args.coff_ratio
        self.coff_cont = explainer_args.coff_cont
        self.init_bias = 0.0
        self.t0 = explainer_args.t0
        self.t1 = explainer_args.t1

        self.init_node_feat_mask(F)
        self.elayers = nn.ModuleList()
        if model_args.model_name == 'gat':
            input_feature = model_args.gat_heads * model_args.gat_hidden * 2
        elif model_args.concate:
            input_feature = int(torch.Tensor(model_args.latent_dim).sum()) * 2
        else:
            input_feature = model_args.latent_dim[-1] * 2
        self.elayers.append(nn.Sequential(nn.Linear(input_feature, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)
        self.ckpt_path = os.path.join('./checkpoint', data_args.dataset_name,
                                      f'PGE_generator_{model_args.model_name}.pth')


    def __set_masks__(self, x, edge_index, edge_mask=None, init="normal"):
        """ Set the weights for message passing """
        (N, F), E = x.size(), edge_index.size(1)
        # std = 0.1
        init_bias = self.init_bias  
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask
            
        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        # self.node_feat_mask = None
        self.edge_mask = None

    @property
    def num_hops(self):
        """ return the number of layers of GNN model """
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __loss__(self, prob, ori_pred, ori_pred_dict, edge_index):
        """
        the pred loss encourages the masked graph with higher probability,
        the size loss encourage small size edge mask,
        the entropy loss encourage the mask to be continuous.
        """
        logit1 = prob + EPS
        logit2 = ori_pred + EPS
        pred_loss = torch.mean(torch.log(logit2) - logit2 * torch.log(logit1))

        # ratio
        edge_mask = torch.sigmoid(self.mask_sigmoid)
        ratio_loss = self.coff_ratio * torch.relu(torch.sum(edge_mask) - explainer_args.ratio_aug * self.nodesize)

        # continuity
        edge_mask = edge_mask * 0.99 + 0.005
        adj_tensor_dense = to_dense_adj(edge_index)
        adj_tensor_dense = torch.squeeze(adj_tensor_dense)
        edge_count = torch.sum(adj_tensor_dense, dim=-1)
        filter_index = np.argwhere(edge_count.cpu().numpy()!=1)
        filter_index = torch.from_numpy(np.squeeze(filter_index))
        noise = 0.001*torch.rand(adj_tensor_dense.shape).to(self.device)
        adj_tensor_dense += noise
        cols = torch.argsort(adj_tensor_dense,dim=-1,descending=True)
        sampled_rows = torch.arange(adj_tensor_dense.shape[0])
        sampled_rows = torch.unsqueeze(sampled_rows,dim=-1).to(self.device)
        edge_count = edge_count.cpu().numpy().tolist()
        sampled0 = None
        for i in range(len(edge_count)):
            for j in range(int(edge_count[i])):
                if sampled0 == None:
                    sampled0 = torch.unsqueeze(torch.cat((sampled_rows[i], torch.unsqueeze(cols[int(edge_count[i]),0],dim=-1)),dim=-1),dim=-1)
                    sampled1 = torch.unsqueeze(torch.cat((sampled_rows[i], torch.unsqueeze(cols[int(edge_count[i]),j],dim=-1)),dim=-1),dim=-1)
                else:
                    sampled0 = torch.cat((sampled0, torch.unsqueeze(torch.cat((sampled_rows[i], torch.unsqueeze(cols[int(edge_count[i]),0],dim=-1)),dim=-1),dim=-1)),dim=-1)
                    sampled1 = torch.cat((sampled1, torch.unsqueeze(torch.cat((sampled_rows[i], torch.unsqueeze(cols[int(edge_count[i]),j],dim=-1)),dim=-1),dim=-1)),dim=-1)
        sample0_score = gather_nd(edge_mask, sampled0.t())
        sample1_score = gather_nd(edge_mask, sampled1.t())
        con_edge_score = gather_nd(edge_mask, edge_index.t())

        cont_loss = self.coff_cont * (torch.mean(-(1.0-sample0_score)*torch.log(1.0-sample1_score) \
            - sample0_score*torch.log(sample1_score)) + torch.mean(con_edge_score*torch.log(con_edge_score) \
                + (1-con_edge_score)*torch.log(1-con_edge_score)))

        if self.mask_features:
            m = self.node_feat_mask.sigmoid()
            node_feat_size_loss = self.coeffs['node_feat_size'] * torch.relu(m.sum() - explainer_args.ratio_aug *m.shape[0])
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            node_feat_ent_loss = self.coeffs['node_feat_ent'] * ent.mean()

        loss = pred_loss + ratio_loss + cont_loss + node_feat_size_loss + node_feat_ent_loss
        # print(pred_loss)
        return loss

    def init_node_feat_mask(self, F):
        """ Set the weights for message passing """
        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F, requires_grad=True, device=self.device) * std)

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """ Sample from the instantiation of concrete distribution when training
        \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})
        """
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def forward(self, inputs, training=None):
        data, embed, tmp = inputs
        edge_index = data.edge_index
        x = data.x
        self.nodesize = embed.shape[0]
        # print(nodesize)
        feature_dim = embed.shape[1]
        f1 = embed.unsqueeze(1).repeat(1, self.nodesize, 1).reshape(-1, feature_dim)
        f2 = embed.unsqueeze(0).repeat(self.nodesize, 1, 1).reshape(-1, feature_dim)

        # using the node embedding to calculate the edge weight
        f12self = torch.cat([f1, f2], dim=-1)
        h = f12self.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        self.mask_sigmoid = values.reshape(self.nodesize, self.nodesize)

        # set the symmetric edge weights
        sym_mask = (self.mask_sigmoid + self.mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_index[0], edge_index[1]]

        # inverse the weights before sigmoid in MessagePassing Module
        edge_mask = inv_sigmoid(edge_mask)
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)

        # the model prediction with edge mask
        if True:
            x = x * self.node_feat_mask.view(1, -1).sigmoid()
        data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
        data.to(self.device)
        outputs = self.model(data)
        return outputs[1], edge_mask, outputs[2]

    def get_model_output(self, x, edge_index, edge_mask=None, **kwargs):
        """ return the model outputs with or without (w/wo) edge mask  """
        self.model.eval()
        self.__clear_masks__()
        if edge_mask is not None:
            self.__set_masks__(x, edge_index, edge_mask.to(self.device))
        data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])            
        data.to(self.device)
        self.model.to_device()
        outputs = self.model(data)

        self.__clear_masks__()
        return outputs

    def train_GC_explanation_network(self, dataset, epoch, mask_features: bool = True):
        tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
        dataset_indices = list(range(len(dataset.to_data_list())))

        self.mask_features = mask_features
        # collect the embedding of nodes
        emb_dict = {}
        ori_pred_dict = {}
        loss = 0
        for gid in dataset_indices:
            data = dataset.to_data_list()[gid]
            _, prob, emb_gnn = self.get_model_output(data.x, data.edge_index)                
            emb_dict[gid] = emb_gnn.data
            ori_pred_dict[gid] = prob.squeeze().argmax(-1).data
            ori_pred = prob.squeeze()
            _, prob, emb_gnn = self.get_model_output(data.x, data.edge_index)
            prob, edge_mask, emb = self.forward((data, emb_gnn, tmp), training=True)
            loss += self.__loss__(prob.squeeze(), ori_pred, ori_pred_dict[gid], data.edge_index)
            if gid == 0:
                emb_batch = emb
            else:
                emb_batch = torch.cat((emb_batch, emb), dim=0)
 
        return loss/len(dataset.to_data_list()), emb_batch

    def get_explanation_network(self, dataset, is_graph_classification=True):
        if os.path.isfile(self.ckpt_path):
            print("fetch network parameters from the saved files")
            state_dict = torch.load(self.ckpt_path)
            self.elayers.load_state_dict(state_dict)
            self.to(self.device)
        elif is_graph_classification:
            self.train_GC_explanation_network(dataset, self.epochs)
        else:
            self.train_NC_explanation_network(dataset)

    def eval_probs(self, x: torch.Tensor, edge_index: torch.Tensor,
                   edge_mask: torch.Tensor=None, **kwargs) -> None:
        outputs = self.get_model_output(x, edge_index, edge_mask=edge_mask)
        return outputs[1].squeeze()

    def explain_edge_mask(self, x, edge_index, **kwargs):
        data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
        data = data.to(self.device)
        with torch.no_grad():
            _, prob, emb = self.get_model_output(data.x, data.edge_index)
            _, edge_mask, _ = self.forward((data, emb, 1.0), training=False)
        return edge_mask

    def get_subgraph(self, node_idx, x, edge_index, y, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        mapping = {int(v): k for k, v in enumerate(subset)}
        subgraph = graph.subgraph(subset.tolist())
        nx.relabel_nodes(subgraph, mapping)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item
        y = y[subset]
        return x, edge_index, y, subset, kwargs

    def train_NC_explanation_network(self, dataset):
        data = dataset[0]
        dataset_indices = torch.where(data.train_mask != 0)[0].tolist()
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)

        # collect the embedding of nodes
        x_dict = {}
        edge_index_dict = {}
        node_idx_dict = {}
        emb_dict = {}
        pred_dict = {}
        with torch.no_grad():
            self.model.eval()
            for gid in dataset_indices:
                x, edge_index, y, subset, _ = \
                    self.get_subgraph(node_idx=gid, x=data.x, edge_index=data.edge_index, y=data.y)
                _, prob, emb = self.get_model_output(x, edge_index)

                x_dict[gid] = x
                edge_index_dict[gid] = edge_index
                node_idx_dict[gid] = int(torch.where(subset == gid)[0])
                pred_dict[gid] = prob[node_idx_dict[gid]].argmax(-1).cpu()
                emb_dict[gid] = emb.data.cpu()

        # train the explanation network
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.epochs):
            loss = 0.0
            acc_list = []
            optimizer.zero_grad()
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
            self.elayers.train()
            for gid in tqdm(dataset_indices):
                pred, edge_mask = self.forward((x_dict[gid], emb_dict[gid], edge_index_dict[gid], tmp), training=True)
                loss_tmp = self.__loss__(pred[node_idx_dict[gid]], pred_dict[gid])
                loss_tmp.backward()
                loss += loss_tmp.item()

                acc_list.append(pred[node_idx_dict[gid]].argmax().item() == data.y[gid])

            optimizer.step()
            accs = torch.stack(acc_list, dim=0)
            acc = np.array(accs).mean()
            print(f'Epoch: {epoch} | Loss: {loss} | Acc : {acc}')
            torch.save(self.elayers.cpu().state_dict(), self.ckpt_path)
            self.elayers.to(self.device)

    def eval_node_probs(self, node_idx: int, x: torch.Tensor,
                        edge_index: torch.Tensor, edge_mask: torch.Tensor, **kwargs):
        probs = self.eval_probs(x=x, edge_index=edge_index, edge_mask=edge_mask, **kwargs)
        return probs[node_idx].squeeze()

    def get_node_prediction(self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor, **kwargs):
        outputs = self.get_model_output(x, edge_index, edge_mask=None, **kwargs)
        return outputs[1][node_idx].argmax(dim=-1)

    def explain_node(self, node_idx, x, edge_index, **kwargs):
        data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
        data = data.to(self.device)
        with torch.no_grad():
            _, prob, emb = self.get_model_output(data.x, data.edge_index)
            _, edge_mask = self.forward((data.x, emb, data.edge_index, 1.0), training=False)
        return edge_mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'
