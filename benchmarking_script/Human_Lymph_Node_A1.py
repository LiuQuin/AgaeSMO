#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('omics1_path',type=str, help='omics1_path')
parser.add_argument('omics2_path',type=str, help='omics2_path')
parser.add_argument('n_cluster',type=int, help='n_cluster')
parser.add_argument('save_path',type=str, help='save_path')
parser.add_argument('save_label',type=str, help='save_label')
parser.add_argument('--epoch',default=600,type=int, help='epoch')
parser.add_argument('--seed',default=2022,type=int, help='seed')
parser.add_argument('--anno',default=False,type=bool, help='epoch')
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
random_seed = args.seed
AgaeSMO_v1.fix_seed(random_seed)

# In[4]:



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# the location of R, which is required for the 'mclust' algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = '/public/home/off_liukunpeng/software/anaconda3/envs/pyg1/lib/R' 


# In[ ]:


# read data
print(args.omics1_path)
adata_omics1 = sc.read_h5ad(args.omics1_path)
adata_omics2 = sc.read_h5ad(args.omics2_path)

adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()


# In[7]:



# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = AgaeSMO_v1.pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)
adata_omics1.obsm['tensor']=adata_omics1[:, adata_omics1.var['highly_variable']].X
# adata_omics1.obsm['tensor']=AgaeSMO_v1.pca(adata_omics1_high, n_comps=100)

adata_omics2 = AgaeSMO_v1.clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = AgaeSMO_v1.pca(adata_omics2, n_comps=adata_omics2.n_vars-1)
# adata_omics2.obsm['tensor']=AgaeSMO_v1.pca(adata_omics2, n_comps=adata_omics2.n_vars-1)
adata_omics2.obsm['tensor']=adata_omics2.X




# In[10]:




# In[11]:


data = AgaeSMO_v1.construct_neighbor_graph(adata_omics1, adata_omics2,6,6,k=6)


# In[12]:


# define model
model = AgaeSMO_v1.Train_AgaeSMO(data, device=device,learning_rate=0.001,epochs=args.epoch,loss_weight=[1,1,1,1])
# print(model.model)
# train model
output = model.train()


# In[13]:


adata = adata_omics1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
adata.obsm['AgaeSMO'] = output['AgaeSMO'].copy()
adata.obsm['alpha'] = output['alpha']
# adata.obsm['alpha_omics1'] = output['alpha_omics1']
# adata.obsm['alpha_omics2'] = output['alpha_omics2']


# In[14]:


n_cluster=args.n_cluster
tool = 'mclust' # mclust, leiden, and louvain  
AgaeSMO_v1.clustering(adata,refine_=False, key='AgaeSMO', add_key='AgaeSMO', n_clusters=n_cluster, method=tool, use_pca=True)# visualization
if args.anno:
    # AgaeSMO_adata=sc.read_h5ad("v1.2_test_result/Dataset11_Human_Lymph_Node_A1.h5ad")
    annotation=pd.read_csv('../data/sg/Dataset11_Human_Lymph_Node_A1/annotation.csv')
    # annotation1=pd.DataFrame(annotation.loc[:,'manual-anno'])
    adata.obs['Ground Truth']=annotation.loc[:,'manual-anno'].to_list()
    indexs=AgaeSMO_v1.supervise_index(adata,"AgaeSMO",'Ground Truth')
    print(indexs)

    ARI=indexs["ARI"]
else:
    ARI=""
import matplotlib.pyplot as plt

fig, ax_list = plt.subplots(1, 2, figsize=(12, 4))
sc.pp.neighbors(adata, use_rep='AgaeSMO', n_neighbors=10)
sc.tl.umap(adata)

sc.pl.umap(adata, color='AgaeSMO', ax=ax_list[0], title='AgaeSMO', s=30, show=False)
sc.pl.embedding(adata, basis='spatial', color='AgaeSMO', ax=ax_list[1], title=f'AgaeSMO:{ARI}', s=50, show=False)

plt.tight_layout(w_pad=0.3)
# plt.show()
plt.savefig(args.save_path+args.save_label+"_AgaeSMO"+f"_{ARI}_{args.seed}"+".jpg")

torch.save(model,args.save_path+args.save_label+".pth")
adata.write(args.save_path+args.save_label+".h5ad")