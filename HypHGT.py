import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn as dglnn
import math

from manifold.layer import HypLinear, HypLayerNorm, HypActivation, HypDropout
from manifold.lorentz import Lorentz

class TransConv(nn.Module):
    def __init__(
        self,
        manifold_in,
        manifold_hidden,
        manifold_out,
        in_dim,
        hidden_dim,
        out_dim,
        device,
        num_relations,
        num_layers=2,
        num_heads=8,
        dropout_rate=0.5,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_act=True,
    ):
        super(TransConv, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.num_layers = num_layers
        self.residual = use_residual
        self.use_act = use_act
        self.use_weight = use_weight
        self.norm_scale = nn.Parameter(torch.ones(()))

        self.device = device
        self.num_relations = num_relations

        self.scale = nn.Parameter(torch.tensor([math.sqrt(hidden_dim)]))
        self.bias = nn.Parameter(torch.zeros(()))

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.power_k = 3.0

        self.Wq = nn.ModuleList()
        self.Wk = nn.ModuleList()
        self.Wv = nn.ModuleList()
        for i in range(num_relations):
            self.Wq.append(nn.ModuleList([HypLinear(self.manifold_hidden[i], self.hidden_dim, self.hidden_dim, self.manifold_hidden[i]) for _ in range(num_heads)]))
            self.Wk.append(nn.ModuleList([HypLinear(self.manifold_hidden[i], self.hidden_dim, self.hidden_dim, self.manifold_hidden[i]) for _ in range(num_heads)]))
            self.Wv.append(nn.ModuleList([HypLinear(self.manifold_hidden[i], self.hidden_dim, self.hidden_dim, self.manifold_hidden[i]) for _ in range(num_heads)]))

        self.epsilon = torch.tensor([0.5])
        for i in range(self.num_relations):
            self.bns.append(HypLayerNorm(self.manifold_hidden[i], self.hidden_dim))
        for i in range(self.num_relations):
            self.fc = nn.ModuleList()
            self.fc.append(HypLinear(self.manifold_in, self.in_dim, self.hidden_dim, self.manifold_hidden[i]))
            self.fc.append(HypLinear(self.manifold_hidden[i], self.hidden_dim, self.hidden_dim, self.manifold_hidden[i]))
            self.fcs.append(self.fc)

        self.positional_encoding = nn.ModuleList()
        for i in range(self.num_relations):
            self.positional_encoding.append(HypLinear(self.manifold_in, self.in_dim, self.hidden_dim, self.manifold_hidden[i]))

        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        self.out_to_hidden = nn.ModuleList()
        self.hidden_to_out = nn.ModuleList()
        for i in range(self.num_relations):
            self.dropout.append(HypDropout(self.manifold_hidden[i], self.dropout_rate))
            self.activation.append(HypActivation(self.manifold_hidden[i], activation=F.leaky_relu))
            self.out_to_hidden.append(HypLinear(self.manifold_out, self.hidden_dim, self.hidden_dim, self.manifold_hidden[i]))
            self.hidden_to_out.append(HypLinear(self.manifold_hidden[i], self.hidden_dim, self.hidden_dim, self.manifold_out))

        self.bn = HypLayerNorm(self.manifold_out, self.hidden_dim)
        self.dr = HypDropout(self.manifold_out, self.dropout_rate)

    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

    def edge_attention(self, edges, i, manifold):
        q = edges.dst["q"][:, :, 1:]
        k = edges.src["k"][:, :, 1:]
        v = edges.src["v"][:, :, 1:]

        phi_qs = (F.leaky_relu(q) + 1e-6) / self.norm_scale.abs()
        phi_ks = (F.leaky_relu(k) + 1e-6) / self.norm_scale.abs()
        phi_vs = (F.leaky_relu(v) + 1e-6) / self.norm_scale.abs()

        phi_qs = self.fp(phi_qs, p=self.power_k)
        phi_ks = self.fp(phi_ks, p=self.power_k)

        k_transpose_v = torch.einsum("nhm,nhd->hmd", phi_ks, phi_vs)  # [H, D, D]
        numerator = torch.einsum("nhm,hmd->nhd", phi_qs, k_transpose_v)  # [N, H, D]
        denominator = torch.einsum("nhd,hd->nh", phi_qs, torch.einsum("nhd->hd", phi_ks))  # [N, H]
        denominator = denominator.unsqueeze(-1)

        attn = numerator / (denominator + 1e-6)  # [N, H, D]

        attn = attn.mean(dim=1)
        attn = self.dropout[i](attn)
        time = ((attn ** 2).sum(dim=-1, keepdim=True) + manifold.k) ** 0.5

        return {"out": self.hidden_to_out[i](torch.cat([time, attn], dim=-1))}

    def forward(self, graph, iter):
        for i, edges in enumerate(graph.canonical_etypes):
            src_type = edges[0]
            etype = edges[1]
            dst_type = edges[2]
            src_feats = graph.nodes[edges[0]].data["h"].to(self.device)
            dst_feats = graph.nodes[edges[2]].data["h"].to(self.device)

            if iter == 0:  # Euclidean feature to hyperbolic space (Lorentz model)
                src_hyp = self.fcs[i][0](src_feats, x_manifold="euc")
                dst_hyp = self.fcs[i][0](dst_feats, x_manifold="euc")
                graph.nodes[src_type].data["residual"] = src_hyp
            else:
                src_hyp = self.out_to_hidden[i](src_feats)
                dst_hyp = self.out_to_hidden[i](dst_feats)
                graph.nodes[src_type].data["residual"] = graph.nodes[src_type].data["h"]

            src_hyp = self.bns[i](src_hyp)
            dst_hyp = self.bns[i](dst_hyp)

            src_hyp = self.dropout[i](self.activation[i](src_hyp))
            dst_hyp = self.dropout[i](self.activation[i](dst_hyp))

            q = torch.stack([W(dst_hyp) for W in self.Wq[i]], dim=1)
            k = torch.stack([W(src_hyp) for W in self.Wk[i]], dim=1)
            v = torch.stack([W(src_hyp) for W in self.Wv[i]], dim=1)

            graph.nodes[dst_type].data["q"] = q
            graph.nodes[src_type].data["k"] = k
            graph.nodes[src_type].data["v"] = v

            graph.apply_edges(lambda edges: self.edge_attention(edges, i, self.manifold_out), etype=etype)

        update_dict = {}
        for i, (src_type, etype, dst_type) in enumerate(graph.canonical_etypes):
            update_dict[(src_type, etype, dst_type)] = (
                fn.copy_e("out", "m"),
                fn.mean("m", "out"),
            )

        graph.multi_update_all(update_dict, cross_reducer="mean")  # 여기 hyperbolic embedding을 단순히 mean 하는 aggregator를 개선할 수 있을 것 같음

        for ntype in graph.ntypes:
            graph.nodes[ntype].data["h"] = self.manifold_out.mid_point(torch.stack((graph.nodes[ntype].data["out"], graph.nodes[ntype].data["residual"]), dim=1))
            graph.nodes[ntype].data["h"] = self.dr(self.bn(graph.nodes[ntype].data["h"]))


class HypHetFormer(nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim, num_relations, num_class, device, is_large):
        super(HypHetFormer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_class = num_class

        self.is_large = is_large

        self.device = device

        self.trans_layer = 3
        self.gnn_heads = 8
        self.trans_heads = 8

        self.manifold_in = Lorentz(k=float(1.0), learnable=True)
        self.manifold_hidden_list = []
        for _ in range(self.num_relations):
            self.manifold_hidden_list.append(Lorentz(k=float(1.0), learnable=True))
        self.manifold_out = Lorentz(k=float(1.0), learnable=True)

        self.feature_mapping = nn.Linear(512, in_dim, bias=False)
        torch.nn.init.xavier_normal_(self.feature_mapping.weight, gain=math.sqrt(2))

        self.TransConv = nn.ModuleList()
        for i, _ in enumerate(range(self.trans_layer)):
            self.TransConv.append((TransConv(self.manifold_in, self.manifold_hidden_list, self.manifold_out, self.in_dim, self.hidden_dim, self.out_dim, self.device, self.num_relations, num_heads=self.trans_heads)))

        self.GraphConv = dglnn.HeteroGraphConv({etype: dglnn.GATv2Conv(in_dim, hidden_dim, num_heads=self.gnn_heads, feat_drop=0.5, activation=F.leaky_relu) for etype in graph.canonical_etypes}, aggregate="mean")
        self.GraphConv2 = dglnn.HeteroGraphConv({etype: dglnn.GATv2Conv(hidden_dim*self.gnn_heads, hidden_dim, num_heads=self.gnn_heads, feat_drop=0.5, activation=F.leaky_relu) for etype in graph.canonical_etypes}, aggregate='mean')
        self.GraphConv3 = dglnn.HeteroGraphConv({etype: dglnn.GATv2Conv(hidden_dim*self.gnn_heads, hidden_dim, num_heads=self.gnn_heads, feat_drop=0.5, activation=F.leaky_relu) for etype in graph.canonical_etypes}, aggregate="mean")

        self.fc3 = nn.Linear(self.hidden_dim, self.out_dim, bias=False)
        self.logits = nn.Linear(self.out_dim, self.num_class, bias=True)

        torch.nn.init.xavier_normal_(self.fc3.weight, gain=math.sqrt(2))
        torch.nn.init.xavier_normal_(self.logits.weight, gain=math.sqrt(2))

        type_linear = []
        for ntype in graph.ntypes:
            type_linear.append([ntype, nn.ModuleList()])
        self.type_linear = nn.ModuleDict(type_linear)

        for ntype in graph.ntypes:
            self.type_linear[ntype] = nn.Linear(hidden_dim * self.gnn_heads, hidden_dim, bias=True)
            torch.nn.init.xavier_normal_(self.type_linear[ntype].weight, gain=math.sqrt(2))

    def forward(self, graph, out_type):  # GNN + Transformer Type 3
        if self.is_large:
            for ntype in graph.ntypes:
                if (graph.nodes[ntype].data["feats"]).size(1) != self.in_dim:
                    graph.nodes[ntype].data["feats"] = self.feature_mapping(graph.nodes[ntype].data["feats"])
                graph.nodes[ntype].data["h"] = graph.nodes[ntype].data["feats"]
        else:
            for ntype in graph.ntypes:
                graph.nodes[ntype].data["h"] = graph.nodes[ntype].data["feats"]

        input_dict = {}
        for ntype in graph.ntypes:
            input_dict[ntype] = graph.nodes[ntype].data["h"]

        # GNN Layer 1
        embedding_dict = self.GraphConv(graph, input_dict)
        concat_embedding_dict = {}
        for key in embedding_dict.keys():
            concat_embedding_dict[key] = embedding_dict[key].view(embedding_dict[key].size(0), -1)

        # GNN Layer 2
        embedding_dict = self.GraphConv2(graph, concat_embedding_dict)
        concat_embedding_dict = {}
        for key in embedding_dict.keys():
            concat_embedding_dict[key] = embedding_dict[key].view(embedding_dict[key].size(0), -1)

        # GNN Layer 3
        embedding_dict = self.GraphConv3(graph, concat_embedding_dict)
        for key in embedding_dict.keys():
            graph.nodes[key].data["gnn"] = self.type_linear[key](embedding_dict[key].view(embedding_dict[key].size(0), -1))

        for i in range(self.trans_layer):
            self.TransConv[i](graph, i)

        lambda_ = 0.5 # 0 : GNN  # 1: Graph transformer
        for key in embedding_dict.keys():
            graph.nodes[key].data["h"] = lambda_ * self.manifold_out.logmap0(graph.nodes[key].data["h"])[..., 1:] + (1 - lambda_) * graph.nodes[key].data["gnn"]

        h = graph.nodes[out_type].data["h"]
        h = self.fc3(h)

        logits = self.logits(h)

        return logits, h
