import os
import torch
import shutil
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader
from pga import PGA
from tqdm import tqdm
from copy import deepcopy
from models import GnnNets, GnnNets_NC
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args, explainer_args

class Graphcl(nn.Module):

    def __init__(self, pgexplainer1, pgexplainer2):
        super(Graphcl, self).__init__()
        self.pgexplainer1 = pgexplainer1
        self.pgexplainer2 = pgexplainer2
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, batch):
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

# train for graph classification
def train_GC():
    # attention the multi-task here
    print('start loading data====================')
    dataset1 = get_dataset(data_args).shuffle()
    dataset2 = deepcopy(dataset1)
    input_dim = dataset1.num_node_features
    if data_args.dataset_name == "zinc_standard_agent":
        output_dim = 200
    else:
        raise ValueError("Invalid dataset name.")
    loader1 = DataLoader(dataset1, batch_size=train_args.batch_size)
    loader2 = DataLoader(dataset2, batch_size=train_args.batch_size)
    _, F = loader1.dataset.data.x.size()
    device = torch.device("cuda:" + str(model_args.device_id)) if torch.cuda.is_available() else torch.device("cpu")
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    pgexplainer_1 = PGA(gnnNets, F=F, epochs=train_args.max_epochs)
    pgexplainer_2 = PGA(gnnNets, F=F, epochs=train_args.max_epochs)
    graphcl = Graphcl(pgexplainer_1, pgexplainer_2)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in graphcl.parameters())))
    graphcl.to(device)
    optimizer = Adam(graphcl.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    print('start training model==================')
    graphcl.train()

    for epoch in range(train_args.max_epochs):
        train_loss_accum = 0
        print("====epoch " + str(epoch))
        for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
            batch1, batch2 = batch
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)

            optimizer.zero_grad()

            loss_pg1, x1 = pgexplainer_1.train_GC_explanation_network(batch1, epoch)
            loss_pg2, x2 = pgexplainer_2.train_GC_explanation_network(batch2, epoch)
            x1 = graphcl.forward_cl(x1, batch1.batch)
            x2 = graphcl.forward_cl(x2, batch2.batch)
            loss_cl = graphcl.loss_cl(x1, x2)
            loss = explainer_args.coff_aug*(loss_pg1 + loss_pg2) + loss_cl
            loss.backward()
            optimizer.step()

            loss = float(loss.detach().cpu().item())
            print(f"|Loss: {loss:.3f} | Contrastive Loss: {loss_cl:.3f} ")
            train_loss_accum += loss
            
        print(f"Train Epoch:{epoch}  | Accum_Loss: {train_loss_accum:.3f} | ")

        if (epoch+1) % (train_args.max_epochs) == 0:
            torch.save(gnnNets.state_dict(), "./models_molCLE/molCLE_cont_" + data_args.dataset_name + '_' + str(train_args.max_epochs) + ".pth")

if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]]()
