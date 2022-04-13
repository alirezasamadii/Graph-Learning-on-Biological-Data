
print("starting")
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from node2vec import Node2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import time
###########################################################################################
################### filtering a specific disease ##########################################
############################################################################################
def filter_disease(disease_gene_association_file):
    columns = ['diseaseName', 'geneSymbol']
    df = pd.read_csv(disease_gene_association_file, sep='\t',usecols=columns)
    is_Neoplasms =  df['diseaseName']=="Neoplasms"
    df_Neoplasms = df[is_Neoplasms]
    #print(df_Neoplasms)
    return df_Neoplasms
###################################################################################################
#### creating graph from ppi and labeling(=attributes) nodes based on gene-disease association  ###
###################################################################################################
def create_G(biogrid_file_name,df_diabetes):
    biogrid = open(biogrid_file_name, 'r')
    lines=biogrid.readlines()
    G=nx.Graph()
    lbls={}
    for line in tqdm(lines):
        x=line.split()
        if df_diabetes['geneSymbol'].str.contains(x[0]).any():
            lbls[x[0]]=1
            G.add_node(x[0],label=1)
        else:
            G.add_node(x[0],label=0)
        if df_diabetes['geneSymbol'].str.contains(x[1]).any(): 
            G.add_node(x[1],label=1)
        else:  
            G.add_node(x[1],label=0)
        G.add_edge(x[0], x[1])
    biogrid.close()
    #print(nx.info(G))
    return (G,lbls)
############################################################################
############################################################################
###################################### BODY  ###############################
############################################################################
############################################################################
print(" (0/6) started ")
disease_gene_association_file="all_gene_disease_associations.tsv"
#disease_gene_association_file="gad.tsv"
df_diabetes=filter_disease(disease_gene_association_file)
print(" (1/6) disease dataframe creation : done")
                   
biogrid_file_name='Biogrid_REDUX.txt'
#biogrid_file_name='test2.txt'
x=create_G(biogrid_file_name,df_diabetes)
G=x[0]
lbls=x[1]
print(" (2/6) Graph creation : done")
# Corpus generation using random walks and Representation Learning using Word2Vec
p_q=[(2,0.1),(2,0.01),(2,0.001),(2,0.0001),
     (20,0.1),(20,0.01),(20,0.001),(20,0.0001),
     (200,0.1),(200,0.01),(200,0.001),(200,0.0001)]
#pick one value from above list and modify list  to avoid for loop and
#p_q=[(2,0.1)]
precisions=[]
for pq in tqdm(p_q):
    node2vec = Node2Vec(G, dimensions=32, workers=1,p=pq[0],q=pq[1]) #=walks
    print(" (3/6) random walks  created. fitting node2vec model... please wait. it will take a few minutes")
    tic = time.perf_counter()
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    toc = time.perf_counter()
    print(f" (4/6)model creation : done in {(toc - tic)/60 :0.1f} minutes")

    X=model.wv.vectors
    y=np.array(list(nx.get_node_attributes(G,"label").values()))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("(5/6) data sepration : done ... classification in progress  : ")

    clf = svm.SVC(kernel='linear') # Create a svm Classifier WITH Linear Kernel
    clf.fit(X_train, y_train)   #Train the model using the training sets
    y_pred = clf.predict(X_test)  #Predict the response for test dataset
    p=metrics.precision_score(y_test, y_pred)
    precisions.append(p)
 
# plotting the points
vals=['(2,0.1)','(2,0.01)','(2,0.001)','(2,0.0001)',
      '(20,0.1)','(20,0.01)','(20,0.001)','(20,0.0001)',
      '(200,0.1)','(200,0.01)','(200,0.001)','(200,0.0001)']
x=[0,25,50,75,100,125,150,175,200,225,250,275]
plt.xticks(x,vals)
plt.plot(p_q, precisions)
#plt.xlabel('(p , q)')
plt.ylabel('precision')
plt.title('precision based on different values of p and q!')
plt.show()
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("F1-Score:",metrics.f1_score(y_test, y_pred))
plot_confusion_matrix(clf, X_test, y_test)  
plt.show()
print(" (6/6) ALL DONE SUCCESSFULLY ")