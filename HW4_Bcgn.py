print("starting")
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T
import networkx as nx
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.metrics import classification_report
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from tensorflow.python.keras.backend import binary_crossentropy
from matplotlib import pyplot
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sys
from tqdm import tqdm, tqdm_notebook
import os.path as osp
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GNNExplainer

##########################################################################################
# custom dataset
class PPIDATASET(InMemoryDataset):
    def __init__(self, transform=None):
        super(PPIDATASET, self).__init__('.', transform, None, None) #pre transform and pre filter: None, we don't need them

        data = Data(edge_index=edge_index)
        
        data.num_nodes = G.number_of_nodes()
        
        # embedding 
        data.x = torch.from_numpy(embeddings).type(torch.float32)
        
        # labels
        y = torch.from_numpy(labels).type(torch.long)
        data.y = y.clone().detach() #removing tensors computational graph for efficency since it is not needed
        
        data.num_classes = 2

        X_train, X_test, y_train, y_test = train_test_split(pd.Series(G.nodes()), pd.Series(labels),test_size=0.30, random_state=42)
        
        n_nodes = G.number_of_nodes()
        
        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask

        self.data, self.slices = self.collate([data])

    # def _download(self):
    #     return

    # def _process(self):
    #     return

    # def __repr__(self):
    #     return '{}()'.format(self.__class__.__name__)
#######################################################################################
# # GCN model with 2 layers 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16) #in feature, out dim
        self.conv2 = GCNConv(16, int(data.num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)    

###########################################################################################
################### filtering a specific disease ##########################################
############################################################################################
def filter_disease(disease_gene_association_file):
    columns = ['diseaseName', 'geneSymbol']
    df = pd.read_csv(disease_gene_association_file, sep='\t',usecols=columns)
    is_Neoplasms =  df['diseaseName']=="Diabetes"
    df_Neoplasms = df[is_Neoplasms]
    #print(df_Neoplasms)
    return df_Neoplasms
###################################################################################################
#### creating graph from ppi and labeling(=attributes) nodes based on gene-disease association  ###
###################################################################################################
def create_G(biogrid_file_name,df_Neoplasms):
    biogrid = open(biogrid_file_name, 'r')
    lines=biogrid.readlines()
    G=nx.Graph()
    for line in tqdm(lines):
        x=line.split()
        if df_Neoplasms['geneSymbol'].str.contains(x[0]).any():
            G.add_node(x[0],label=1)
        else:
            G.add_node(x[0],label=0)
        if df_Neoplasms['geneSymbol'].str.contains(x[1]).any(): 
            G.add_node(x[1],label=1)
        else:  
            G.add_node(x[1],label=0)
        G.add_edge(x[0], x[1])
        G.add_edge(x[0], x[0]) # i decided to include an edge from node to itself to have adj matrix with himself included
        G.add_edge(x[1], x[1])
    biogrid.close()
    G=nx.convert_node_labels_to_integers(G)
    #print(nx.info(G))
    return G
#####################################################################
def train():
  model.train()
  optimizer.zero_grad()
  #negative log likelihood, we need it after a softmax output activation function
  #we also tell to compute the loss using the  mask (so, on the train samples) 
  F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward() 
  optimizer.step()
  
  # comment the following for avoiding eval during training
  model.eval()
  logits = model(data)
  train_mask = data['train_mask']
  train_pred = logits[train_mask].max(1)[1]
  train_acc = train_pred.eq(data.y[train_mask]).sum().item() / train_mask.sum().item() #eq: elementwise equality

  return train_acc
  
####################################################################################
@torch.no_grad()
def test():
  model.eval()
  logits = model(data)

  # uncomment the following if you want to eval on the train and test together
  # train_mask = data['train_mask']
  # train_pred = logits[train_mask].max(1)[1]
  # train_acc = train_pred.eq(data.y[train_mask]).sum().item() / train_mask.sum().item()

  test_mask = data['test_mask']
  test_pred = logits[test_mask].max(1)[1]
  test_acc = test_pred.eq(data.y[test_mask]).sum().item() / test_mask.sum().item()

  return test_acc
############################################################################
############################################################################
###################################### BODY  ###############################
############################################################################
############################################################################
print(" (0/6) started ")
#disease_gene_association_file="all_gene_disease_associations.tsv"
disease_gene_association_file="gad.tsv"
df_Neoplasms=filter_disease(disease_gene_association_file)
print(" (1/6) disease dataframe creation : done")
                   
#biogrid_file_name='Biogrid_REDUX.txt'
biogrid_file_name='test2.txt'
G=create_G(biogrid_file_name,df_Neoplasms)
print(" (2/6) Graph creation : done")
# retrieve the labels for each node
labels = np.asarray([G.nodes[i]['label'] != 0 for i in G.nodes]).astype(np.int64)
# create edge index. We need to have data as previously shown. We can exploit networkX and scipy for that 
adj = nx.to_scipy_sparse_matrix(G).tocoo() #coordinate format
#create edge index in the proper way
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)
# using degree as embedding. For simplicity, the feature vector describing the 
# will be just its degree, which is enough for us
embeddings = np.array(list(dict(G.degree()).values()))
# normalizing degree values
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
embeddings = scale.fit_transform(embeddings.reshape(-1,1))
dataset = PPIDATASET()
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data =  data.to(device)

model = Net().to(device) 
torch.manual_seed(42)

#optimizer_name = "Adam"
lr = 1e-1
#optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 200
for epoch in tqdm_notebook(range(1, epochs)):
  train_acc = train()

  

test_acc = test()

print('#' * 70)
print('Train Accuracy: %s' % train_acc)
print('Test Accuracy: %s' % test_acc)
print('#' * 70)


