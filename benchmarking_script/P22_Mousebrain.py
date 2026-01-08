#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('omics1_path',type=str, help='omics1_path')
parser.add_argument('omics2_path',type=str, help='omics2_path')
parser.add_argument('n_cluster',type=int, help='omics2_path')
parser.add_argument('save_path',type=str, help='omics2_path')
parser.add_argument('save_label',type=str, help='omics2_path')
args = parser.parse_args()

import scanpy as sc
import torch
import os
import pandas as pd
os.chdir("/public/home/off_liukunpeng/project/11_cluster_problem/AgaeSMO")


# In[2]:


import sys
sys.path.append("/public/home/off_liukunpeng/project/11_cluster_problem/AgaeSMO")


# In[3]:


import community as louvain
import AgaeSMO as AgaeSMO_v1
random_seed = 2022
AgaeSMO_v1.fix_seed(random_seed)

# In[4]:



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# the location of R, which is required for the 'mclust' algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = '/public/home/off_liukunpeng/software/anaconda3/envs/pyg1/lib/R' 


# In[ ]:


# read data
# file_fold = '/public/home/off_liukunpeng/project/11_cluster_problem/data/sma/V11L12-109_B1/' #please replace 'file_fold' with the download path
print(args.omics1_path)
adata_omics1 = sc.read_h5ad(args.omics1_path)
adata_omics2 = sc.read_h5ad(args.omics2_path)
print(adata_omics1.X.shape)
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()
print(adata_omics1.X.shape)
# In[7]:

# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = AgaeSMO_v1.pca(adata_omics1_high, n_comps=50)
adata_omics1.obsm['tensor']=adata_omics1[:, adata_omics1.var['highly_variable']].X
print(adata_omics1[:, adata_omics1.var['highly_variable']].X.shape)
# adata_omics1.obsm['tensor']= AgaeSMO_v1.pca(adata_omics1_high, n_comps=50)
#
print(adata_omics1.X.shape)

adata_omics2 = adata_omics2[adata_omics1.obs_names].copy() # .obsm['X_lsi'] represents the dimension reduced feature
# if 'X_lsi' not in adata_omics2.obsm.keys():
#     sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
#     AgaeSMO_v1.lsi(adata_omics2, use_highly_variable=False, n_components=51)
# adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
# adata_omics2.obsm['tensor'] = adata_omics2.obsm['X_lsi'].copy()

# In[10]:

sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
adata_omics2 = AgaeSMO_v1.clr_normalize_each_cell(adata_omics2)
sc.pp.normalize_total(adata_omics2, target_sum=1e4)
sc.pp.log1p(adata_omics2)
sc.pp.scale(adata_omics2)

adata_omics2_high =  adata_omics2[:, adata_omics2.var['highly_variable']]

adata_omics2.obsm['feat'] = AgaeSMO_v1.pca(adata_omics2_high, n_comps=50)
adata_omics2.obsm['tensor']= adata_omics2[:, adata_omics2.var['highly_variable']].X
print(adata_omics2[:, adata_omics2.var['highly_variable']].X.shape[0])
# adata_omics2.obsm['tensor']=  AgaeSMO_v1.pca(adata_omics2, n_comps=50)


# In[11]:


data = AgaeSMO_v1.construct_neighbor_graph(adata_omics1, adata_omics2,4,4,k=4)


# In[12]:


# define model

model = AgaeSMO_v1.Train_AgaeSMO(data, device=device,epochs=2000,learning_rate=0.001,weight_decay=0.0001,dim_output=60)

# train model
output = model.train()


# In[13]:

adata_omics1.obsm['tensor']=adata_omics1.obsm['spatial']
adata = adata_omics1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
adata.obsm['AgaeSMO'] = output['AgaeSMO'].copy()
adata.obsm['alpha'] = output['alpha']
# adata.obsm['alpha_omics1'] = output['alpha_omics1']
# adata.obsm['alpha_omics2'] = output['alpha_omics2']

print(adata.X.shape)
# In[14]:


n_cluster=args.n_cluster
tool = 'mclust' # mclust, leiden, and louvain  
AgaeSMO_v1.clustering(adata,refine_=True, key='AgaeSMO', add_key='AgaeSMO', n_clusters=n_cluster, method=tool, use_pca=True)# visualization

from sklearn import metrics
def supervise_index(adata,predict,real_key):
    
    ARI = metrics.adjusted_rand_score(adata.obs[real_key],adata.obs[predict])
    NMI = metrics.normalized_mutual_info_score(adata.obs[real_key],adata.obs[predict])
    FMS = metrics.fowlkes_mallows_score(adata.obs[real_key],adata.obs[predict])
    AMI = metrics.adjusted_mutual_info_score(adata.obs[real_key],adata.obs[predict])
    HMG = metrics.homogeneity_score(adata.obs[real_key],adata.obs[predict])
    VMS = metrics.v_measure_score(adata.obs[real_key],adata.obs[predict])
    MIS = metrics.mutual_info_score(adata.obs[real_key],adata.obs[predict])
    return {"ARI":ARI,
            "NMI":NMI,
            "FMS":FMS,
            "AMI":AMI,
            "HMG":HMG,
            "VMS":VMS,
            'MIS':MIS}
# adata=sc.read_h5ad("./ATAC/Dataset7_Mouse_Brain_ATAC.h5ad")
annotation=pd.read_csv("../data/sg/Dataset7_Mouse_Brain_ATAC/41467_2024_55204_MOESM5_ESM.csv")
adata.obs["groundtruth"]=list(annotation["LayerName"])

ari=supervise_index(adata,"AgaeSMO","groundtruth")
ari=ari["ARI"]


import matplotlib.pyplot as plt
fig, ax_list = plt.subplots(1, 2, figsize=(12, 4))
sc.pp.neighbors(adata, use_rep='AgaeSMO', n_neighbors=30)
sc.tl.umap(adata)

sc.pl.umap(adata, color='AgaeSMO', ax=ax_list[0], title='AgaeSMO', s=30, show=False)
sc.pl.embedding(adata, basis='spatial', color='AgaeSMO', ax=ax_list[1], title='AgaeSMO', s=50, show=False)

plt.tight_layout(w_pad=0.3)
plt.show()
plt.savefig(args.save_path+args.save_label+f"_AgaeSMO_{ari}"+".jpg")




print(ari)
# In[17]:

plt.clf()
# AgaeSMO_v1.plot_weight_value(adata.obsm['alpha'],1,show=False)
# plt.savefig(args.save_path+args.save_label+"_weight"+".jpg")
adata.write(args.save_path+args.save_label+".h5ad")