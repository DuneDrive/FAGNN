import torch
import torch.nn as nn
import torch.nn.functional as F
from topk_pool_hsm import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, TransformerConv
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops
from torch_sparse import spspmm
from net.braingraphconv import MyNNConv

class CNN_2(nn.Module):
    def __init__(self, indim, ratio, nclass, batchSize, k=6, R=332):
        super(CNN_2, self).__init__()

        self.num_channels = 10
        self.kernel_size = 3
        self.conv1 = nn.Conv2d(1, self.num_channels, self.kernel_size, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, self.kernel_size, stride=(2,2))
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, self.kernel_size, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(self.num_channels)
        self.fc1 = nn.Linear(600, 128)  # Adjust input size accordingly
        self.fc2 = nn.Linear(128, 32)

    def forward(self, behav_pre):
        behav = behav_pre.type(torch.cuda.FloatTensor)
        behav = behav.view(behav.size(0), 1, behav.size(1), behav.size(2))
        behav = F.relu(self.bn1(self.conv1(behav)))
        behav = F.relu(self.bn2(self.conv2(behav)))
        behav = F.relu(self.bn3(self.conv3(behav)))
        behav = torch.flatten(behav, start_dim=1)
        behav = F.sigmoid(self.fc2(self.fc1(behav)))
        return behav


class CNN_1(nn.Module):
    def __init__(self, indim, ratio, nclass, batchSize, k=6, R=332):
        super(CNN_1, self).__init__()

        self.num_channels = 16
        self.kernel_size = 3
        self.att_ini = nn.Linear(4, 4)
        self.conv1 = nn.Conv1d(1, self.num_channels, self.kernel_size, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(self.num_channels)
        self.conv2 = nn.Conv1d(self.num_channels, self.num_channels, self.kernel_size, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(self.num_channels)
        self.fc = nn.Linear(self.num_channels * 4, 16)  # Adjust input size accordingly

    def forward(self, pheno):
        pheno = pheno.type(torch.cuda.FloatTensor)
        pheno_att = F.sigmoid(self.att_ini(pheno))
        pheno = pheno * pheno_att
        pheno = pheno.view(pheno.size(0), 1, pheno.size(1))
        pheno = F.relu(self.bn1(self.conv1(pheno)))
        pheno = torch.flatten(pheno, start_dim=1)
        pheno = F.sigmoid(self.fc(pheno))
        return pheno, pheno_att


class GraphTransformer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphTransformer, self).__init__()
        self.head_num = 5
        self.tconv1 = TransformerConv(in_channels, out_channels, heads=self.head_num, dropout=0.6, concat=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.lin1 = nn.Linear(out_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.tconv1(x, edge_index)
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        return x


class QuadrantAttentionModel(nn.Module):
    def __init__(self):
        super(QuadrantAttentionModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=166, num_heads=2)
    
    def forward(self, x):
        x = x.view(-1, 332, 332)
        quadrants = [x[:, :166, :166].transpose(0, 1),
                     x[:, :166, 166:332].transpose(0, 1),
                     x[:, 166:332, :166].transpose(0, 1),
                     x[:, 166:332, 166:332].transpose(0, 1)]
        for i in range(4):
            quadrants[i], _ = self.attention(quadrants[i], quadrants[i], quadrants[i])
            quadrants[i] = quadrants[i].transpose(0, 1)
        top_half = torch.cat((quadrants[0], quadrants[1]), dim=2)
        bottom_half = torch.cat((quadrants[2], quadrants[3]), dim=2)
        out = torch.cat((top_half, bottom_half), dim=1).view(-1, 332)
        return out


class GNN(nn.Module):
    def __init__(self, indim, ratio, nclass, batchSize, k=6, R=332):
        super(GNN, self).__init__()
        self.indim = indim
        self.k = k
        self.R = R
        self.dim1 = 332
        self.dim2 = 332
        self.dim3 = 128

        self.conv1 = MyNNConv(self.indim, self.dim1, self.create_sequential(self.R, self.k, self.dim1 * self.indim), normalize=False)
        self.pool1 = TopKPooling(self.dim1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.conv2 = MyNNConv(self.dim1, self.dim2, self.create_sequential(self.R, self.k, self.dim2 * self.dim1), normalize=False)
        self.pool2 = TopKPooling(self.dim2, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)

        self.fc1 = nn.Linear((self.dim1+self.dim2)*2, self.dim2)
        self.bn1 = nn.BatchNorm1d(self.dim2)
        self.fc2 = nn.Linear(self.dim2, self.dim3)
        self.bn2 = nn.BatchNorm1d(self.dim3)
        self.fc3 = nn.Linear(self.dim3, nclass)

    def create_sequential(self, R, k, output_dim):
        return nn.Sequential(nn.Linear(R, k, bias=False), nn.ReLU(inplace=False), nn.Linear(k, output_dim))

    def forward(self, x, edge_index, batch, edge_attr, pos, pheno):
        x = self.conv1(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.fc3(x)

        return x,self.pool1.weight,self.pool2.weight, score0, score1, score2, score_extra

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        return edge_index, edge_weight


class Combined_layer(nn.Module):
    def __init__(self, nclass, input_size):
        super(Combined_layer, self).__init__()
        dim_size1 = 64
        self.fc1 = nn.Linear(input_size, dim_size1)
        self.bn1 = nn.BatchNorm1d(dim_size1)
        self.fc2 = nn.Linear(dim_size1, nclass)

    def forward(self, x_combined):
        x_combined = F.relu(self.fc1(x_combined))
        x_combined = self.bn1(x_combined)
        x_combined = self.fc2(x_combined)
        return x_combined


class FAGNN(nn.Module):
    def __init__(self, indim, ratio, nclass, batchSize, k=6, R=332):
        super(FAGNN, self).__init__()
        self.cnn_1 = CNN_1(indim, ratio, nclass, batchSize)
        self.cnn_2 = CNN_2(indim, ratio, nclass, batchSize)
        self.gt = QuadrantAttentionModel()
        self.gnn = GNN(indim, ratio, nclass, batchSize, k, R)
        self.last_layer = Combined_layer(nclass, 176)  # Adjust combined size accordingly

    def forward(self, x, edge_index, batch, edge_attr, pos, pheno, behav):
        edge_scores = self.gt(x)
        weighted_x = x * edge_scores
        x_gnn, pool1_weight, pool2_weight, score0, score1, score2, score_extra = self.gnn(weighted_x, edge_index, batch, edge_attr, pos, pheno)
        xp_1, _ = self.cnn_1(pheno)
        xp_2 = self.cnn_2(behav)
        x_combined = torch.cat((x_gnn, xp_1, xp_2), dim=1)
        x_combined = self.last_layer(x_combined)
        return x_combined, pool1_weight, pool2_weight, torch.sigmoid(score0).view(x_gnn.size(0),-1), torch.sigmoid(score1).view(x_gnn.size(0),-1), torch.sigmoid(score2).view(x_gnn.size(0),-1), torch.sigmoid(score_extra).view(x_gnn.size(0),-1), edge_scores


