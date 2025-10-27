import torch
import torch.optim as optim
import torch.nn.functional as F

import dgl
import time
import networkx as nx
import pickle as pkl
import numpy as np

from collections import defaultdict

from utils.pytorchtools import EarlyStopping
from utils.tools import svm_test

from HypHGT import HypHetFormer


def load_dataset():
    with open("./datasets/"+dataset+"/node_features.pkl", "rb") as f:
        node_features = pkl.load(f)
    with open("./datasets/"+dataset+"/node_type", "rb") as f:
        node_type = pkl.load(f)
        # sorted_dict = sorted(node_type.items(), key = lambda item: item[0], reverse = True)
        # print(sorted_dict)
    movie_global_idx = list(range(0, 4661))  # 0~4660 : movie nodes
    director_global_idx = list(range(4661, 6931))  # 4661~6930 : director nodes
    actor_global_idx = list(range(6931, 12772))  # 6931~12771 # actor nodes

    num_nodes_dict = {
        'M': len(movie_global_idx),
        'D': len(director_global_idx),
        'A': len(actor_global_idx)
    }
    # index mapping dictionary
    gid2lid = {}
    for i, gid in enumerate(movie_global_idx):
        gid2lid[gid] = ('M', i)
    for i, gid in enumerate(director_global_idx):
        gid2lid[gid] = ('D', i)
    for i, gid in enumerate(actor_global_idx):
        gid2lid[gid] = ('A', i)

    edge_list = []
    for i in range(4):
        edge_list.append(
            list(nx.read_edgelist("./datasets/" + dataset + "/edge_" + str(i), create_using=nx.DiGraph(), nodetype=int).edges))

    def add_edges(global_edge_list, srctype, etype, dsttype):
        for src_gid, dst_gid in global_edge_list:
            src_type, src_lid = gid2lid[src_gid]
            dst_type, dst_lid = gid2lid[dst_gid]
            assert src_type == srctype and dst_type == dsttype
            edge_dict[(srctype, etype, dsttype)][0].append(src_lid)
            edge_dict[(srctype, etype, dsttype)][1].append(dst_lid)

    edge_dict = defaultdict(lambda: ([], []))
    add_edges(edge_list[0], 'M', 'M-D', 'D')
    add_edges(edge_list[1], 'D', 'D-M', 'M')
    add_edges(edge_list[2], 'M', 'M-A', 'A')
    add_edges(edge_list[3], 'A', 'A-M', 'M')

    G = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
    G.node_dict = {}
    G.edge_dict = {}
    for ntype in G.ntypes:
        G.node_dict[ntype] = len(G.node_dict)
    for etype in G.etypes:
        G.edge_dict[etype] = len(G.edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype]

    # initialize node features
    for ntype in G.ntypes:
        if ntype == 'M':
            G.nodes[ntype].data['feats'] = torch.Tensor(node_features[movie_global_idx])
        elif ntype == 'D':
            G.nodes[ntype].data['feats'] = torch.Tensor(node_features[director_global_idx])
        elif ntype == 'A':
            G.nodes[ntype].data['feats'] = torch.Tensor(node_features[actor_global_idx])
        else:
            raise print("node-type error has been occured")

    return G

def run_model_IMDB():
    micro_f1_list = []
    macro_f1_list = []

    for iter in range(5):
        net = HypHetFormer(hete_G, hete_G.nodes['M'].data['feats'].size(1), hidden_dim, out_dim, num_relations, num_class, device=device, is_large=False)
        net.to(device)
        optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-6)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=epochs, max_lr=5e-3, pct_start=0.1, anneal_strategy='cos', cycle_momentum=False)
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path="checkpoint/checkpoint_IMDB_" + str(train_percent) + ".pt")
        for epoch in range(epochs):
            t0 = time.time()
            net.train()
            optimizer.zero_grad()
            logits, embeddings = net(hete_G, out_type)
            log_p = F.log_softmax(logits[train_indices], 1)
            train_loss = F.nll_loss(log_p, train_labels)
            train_loss.backward()
            optimizer.step()
            scheduler.step()

            net.eval()
            with torch.no_grad():
                logits, embeddings = net(hete_G, out_type)
                log_p = F.log_softmax(logits[val_indices], 1)
                val_loss = F.nll_loss(log_p, val_labels)
            t1 = time.time()
            if epoch % 1 == 0:
                print('Epoch {} | Train_Loss {:.4f} | Val_Loss {:.4f} | time(epoch/sec) {}'.format(epoch, train_loss.item(), val_loss.item(), t1 - t0))
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print("Early Stopping!!!")
                break;
        net.load_state_dict(torch.load("checkpoint/checkpoint_IMDB_" + str(train_percent) + ".pt"))
        net.eval()
        with torch.no_grad():
            logits, embeddings = net(hete_G, out_type)
        macro_f1, micro_f1 = svm_test(embeddings[test_indices].cpu().numpy(), test_labels.cpu().numpy())
        macro_f1_list.append(macro_f1)
        micro_f1_list.append(micro_f1)

    print("----------------------------------------------------------------------")
    print('SVM tests summary')
    print('Micro-F1: {:.6f}~{:.6f}'.format(np.mean(micro_f1_list), np.std(micro_f1_list)))
    print('Macro-F1: {:.6f}~{:.6f}'.format(np.mean(macro_f1_list), np.std(macro_f1_list)))
    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    dataset = "IMDB"
    hidden_dim = 128
    out_dim = 64
    num_class = 3
    num_layer = 4
    num_relations = 4
    learning_rate = 0.0001
    weight_decay = 0.0005
    out_type = 'M'

    epochs = 300
    patience = epochs

    train_percent_list = [60]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    hete_G = load_dataset().to(device)

    for train_percent in train_percent_list:
        with open('datasets/' + dataset + '/labels_' + str(train_percent) + '.pkl', 'rb') as f:
            labels = pkl.load(f)

        train_indices = ((np.array(labels[0], dtype=np.int64))[:, 0]).tolist()
        val_indices = ((np.array(labels[1], dtype=np.int64))[:, 0]).tolist()
        test_indices = ((np.array(labels[2], dtype=np.int64))[:, 0]).tolist()

        train_labels = torch.LongTensor(labels[0][:, 1].tolist()).to(device)
        val_labels = torch.LongTensor(labels[1][:, 1].tolist()).to(device)
        test_labels = torch.LongTensor(labels[2][:, 1].tolist()).to(device)

        run_model_IMDB()


