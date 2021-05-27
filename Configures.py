import os
import torch
from tap import Tap
from typing import List


class DataParser(Tap):
    dataset_name: str = 'zinc_standard_agent'
    dataset_dir: str = './datasets'
    random_split: bool = False
    scaffold_split: bool = True
    data_split_ratio: List = [0.8, 0.1, 0.1]   # the ratio of training, validation and testing set for random split
    seed: int = 1


class GATParser(Tap):           # hyper-parameter for gat model
    gat_dropout: float = 0.6    # dropout in gat layer
    gat_heads: int = 10         # multi-head
    gat_hidden: int = 10        # the hidden units for each head
    gat_concate: bool = True    # the concatenation of the multi-head feature
    num_gat_layer: int = 3      # the gat layers


class ModelParser(GATParser):
    device_id: int = 2
    model_name: str = 'gin'
    checkpoint: str = './checkpoint'
    concate: bool = False                     # whether to concate the gnn features before mlp
    latent_dim: List[int] = [300, 300, 300]   # the hidden units for each gnn layer
    readout: 'str' = 'sum'                    # the graph pooling method
    mlp_hidden: List[int] = []                # the hidden units for mlp classifier
    gnn_dropout: float = 0.0                  # the dropout after gnn layers
    dropout: float = 0.5                      # the dropout after mlp layers
    adj_normlize: bool = True                 # the edge_weight normalization for gcn conv
    emb_normlize: bool = False                # the l2 normalization after gnn layer
    model_path: str = ""                      # default path to save the model
    input_model_file: str = ""                # default path to load pre-trained model

    def process_args(self) -> None:
        # self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda', self.device_id)
        else:
            pass

        if not self.model_path:
            self.model_path = os.path.join(self.checkpoint,
                                           DataParser().parse_args(known_only=True).dataset_name,
                                           f"{self.model_name}_finetune_best.pth")


class TrainParser(Tap):
    learning_rate: float = 1e-5
    batch_size: int = 32
    weight_decay: float = 0.0
    max_epochs: int = 5
    save_epoch: int = 10
    early_stopping: int = 100


class ExplainerParser(Tap):
    t0: float = 5.0                   # temperature denominator
    t1: float = 1.0                   # temperature numerator
    coff_ratio: float = 0.1           # constrains on mask size
    coff_cont: float = 0.1            # constrains on smooth and continuous mask
    coff_ent: float = 0.1             # constrains on discretized mask
    coff_aug: float = 1e-5            # augmentation coefficient
    ratio_aug: float = 0.75           # augmentation ratio


data_args = DataParser().parse_args(known_only=True)
model_args = ModelParser().parse_args(known_only=True)
train_args = TrainParser().parse_args(known_only=True)
explainer_args = ExplainerParser().parse_args(known_only=True)


import random
import numpy as np
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
