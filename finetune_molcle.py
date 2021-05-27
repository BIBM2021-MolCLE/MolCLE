import os
import sys
import torch
torch.set_printoptions(profile="full")
import shutil
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import torch.nn as nn
from torch.optim import Adam
from models import GnnNets, GnnNets_NC
from load_dataset import get_dataset, get_dataloader
from Configures_finetune import data_args, train_args, model_args
from sklearn.metrics import roc_auc_score

from tensorboardX import SummaryWriter

# train for graph classification
def train_GC():
    # attention the multi-task here
    print('start loading data====================')
    dataset = get_dataset(data_args)
    input_dim = dataset.num_node_features
    # output_dim = int(dataset.num_classes)
    #Bunch of classification tasks
    if data_args.dataset_name == "tox21":
        output_dim = 12
    elif data_args.dataset_name == "hiv":
        output_dim = 1
    elif data_args.dataset_name == "pcba":
        output_dim = 128
    elif data_args.dataset_name == "muv":
        output_dim = 17
    elif data_args.dataset_name == "bace":
        output_dim = 1
    elif data_args.dataset_name == "bbbp":
        output_dim = 1
    elif data_args.dataset_name == "toxcast":
        output_dim = 617
    elif data_args.dataset_name == "sider":
        output_dim = 27
    elif data_args.dataset_name == "clintox":
        output_dim = 2
    elif data_args.dataset_name == "zinc_standard_agent":
        output_dim = 1
    else:
        raise ValueError("Invalid dataset name.")
    dataloader = get_dataloader(dataset, data_args, train_args)

    print('start training model==================')
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    if not model_args.input_model_file == "":
        print('Loading pre-trained model...')
        model_dict = gnnNets.state_dict()
        save_model = torch.load(model_args.input_model_file)
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        print(state_dict.keys())
        model_dict.update(state_dict)
        gnnNets.load_state_dict(model_dict)
        # gnnNets.update_state_dict(torch.load(model_args.input_model_file))
    gnnNets.to_device()
    criterion = nn.BCEWithLogitsLoss(reduction = "none")
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}")

    best_acc = 0.0
    best_roc = 0.0
    data_size = len(dataset)
    print(f'The total num of dataset is {data_size}')

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
        os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"

    early_stop_count = 0
    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        gnnNets.train()
        for batch in dataloader['train']:
            logits, probs, _ = gnnNets(batch)
            _, prediction = torch.max(logits, -1)
            y = batch.y.view(logits.shape).to(torch.float64)
            # loss = criterion(logits, batch.y)

            #Whether y is non-null or not.
            is_valid = y >= 0
            # print(is_valid)
            #Loss matrix
            loss_mat = criterion(logits.double(), y)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                
            optimizer.zero_grad()
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss.backward()

            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            loss_list.append(loss.item())
            # print(prediction.size(), batch.y.size())
            # acc.append(prediction.eq(batch.y).cpu().numpy())
        train_roc = evaluate_roc(gnnNets, dataloader['train'])

        # report train msg
        print(f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | "
            #   f"Acc: {np.concatenate(acc, axis=0).mean():.3f} | "
              f"ROC: {train_roc:.3f}")

        # report eval msg
        eval_state = evaluate_GC(dataloader['eval'], gnnNets, criterion)
        eval_roc = evaluate_roc(gnnNets, dataloader['eval'])
        print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | ROC: {eval_roc:.3f}")

        # only save the best model
        is_best = (eval_roc > best_roc)

        if eval_roc > best_roc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            # best_acc = eval_state['acc']
            best_roc = eval_roc
            early_stop_count = 0
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, is_best)

    print(f"The best validation ROC is {best_roc}.")
    # report test msg
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_finetune_best.pth'))
    gnnNets.update_state_dict(checkpoint['net'])
    test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
    test_roc = evaluate_roc(gnnNets, dataloader['test'])
    print(f"Test: | Loss: {test_state['loss']:.3f} | ROC: {test_roc:.3f}")


def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            logits, probs, _ = gnnNets(batch)
            _, prediction = torch.max(logits, -1)
            y = batch.y.view(logits.shape).to(torch.float64)
            #Whether y is non-null or not.
            is_valid = y >= 0
            #Loss matrix
            loss_mat = criterion(logits.double(), y)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)

            ## record
            loss_list.append(loss.item())
            # acc.append(prediction.eq(batch.y).cpu().numpy())

        eval_state = {'loss': np.average(loss_list)}
                    #   'acc': np.concatenate(acc, axis=0).mean()}

    return eval_state

def evaluate_roc(gnnNets, loader):
    gnnNets.eval()
    y_true = []
    y_scores = []

    for batch in loader:
        # batch = batch.to(device)

        with torch.no_grad():
            logits, probs, _ = gnnNets(batch)
            # _, pred = torch.max(logits, -1)

        y_true.append(batch.y.view(logits.shape))
        y_scores.append(logits)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    # print(y_true.shape[1])

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            # y_true = torch.where(y_true==1, loss_mat, torch.zeros(y_true.shape).to(y_true.device).to(y_true.dtype))
            is_valid = y_true[:,i] >= 0
            # print(y_true[is_valid,i])
            roc_list.append(roc_auc_score(y_true[is_valid,i], y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]

def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, probs, _ = gnnNets(batch)
            _, prediction = torch.max(logits, -1)
            y = batch.y.view(logits.shape).to(torch.float64)
            #Whether y is non-null or not.
            is_valid = y >= 0
            #Loss matrix
            loss_mat = criterion(logits.double(), y)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)

            # record
            loss_list.append(loss.item())
            # acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)

    test_state = {'loss': np.average(loss_list)}
                #   'acc': np.average(np.concatenate(acc, axis=0).mean())}

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, pred_probs, predictions


def save_best(ckpt_dir, epoch, gnnNets, model_name, is_best):
    print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch
        # 'acc': eval_acc
    }
    pth_name = f"{model_name}_finetune_latest.pth"
    best_pth_name = f'{model_name}_finetune_best.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    gnnNets.to_device()


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]]()
