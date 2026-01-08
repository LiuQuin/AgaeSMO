#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('omics1_path',type=str, help='omics1_path')
parser.add_argument('count_file',type=str, help='count_file')
# parser.add_argument('annotation_path',type=str, help='annotation_path')
parser.add_argument('n_cluster',type=int, help='n_cluster')
parser.add_argument('save_path',type=str, help='save_path')
parser.add_argument('save_label',type=str, help='save_label')
parser.add_argument('--epoch',default=600,type=int, help='epoch')
parser.add_argument('--seed',default=2022,type=int, help='seed')
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
import matplotlib.pyplot as plt

import community as louvain
import AgaeSMO as AgaeSMO_v1
random_seed = args.seed
AgaeSMO_v1.fix_seed(random_seed)

# In[3]:

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# the location of R, which is required for the 'mclust' algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = '/public/home/off_liukunpeng/software/anaconda3/envs/pyg1/lib/R' 
# read data

adata_omics1 = sc.read_visium(path=args.omics1_path,
                              count_file=args.count_file,
                              source_image_path="spatial")


adata_omics1.var_names_make_unique()


# In[4]:


adata_omics1


# In[5]:


# RNA
adata_omics1.var["mt"] = adata_omics1.var_names.str.startswith("MT-")
adata_omics1.var["ercc"] = adata_omics1.var_names.str.startswith("ERCC-")
sc.pp.calculate_qc_metrics(adata_omics1, qc_vars=["mt","ercc"], inplace=True)
sc.pp.filter_genes(adata_omics1, min_cells=3)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)



adata=adata_omics1.copy()
his_key=list(adata.uns["spatial"].keys())[0]
#int spot xy in his
adata.obsm["spatial_px"]=adata.obsm["spatial"]*adata.uns["spatial"][his_key]["scalefactors"]["tissue_hires_scalef"]
adata.obsm["spatial_px"]=adata.obsm["spatial_px"].astype(int)
# plot_spot_his(adata.uns["spatial"][his_key]["images"]["hires"],adata.obsm["spatial_px"])
#calculate box xy
r=8
adata=AgaeSMO_v1.calculate_box(adata,adata.obsm["spatial_px"],r)
adata.obsm['patch']=AgaeSMO_v1.his_path(adata,adata.uns["spatial"][his_key]["images"]["hires"])
adata.obsm["patch_flattern"]=adata.obsm["patch"].reshape(adata.obsm["patch"].shape[0],-1)
AgaeSMO_v1.check_patch_his(adata,his_key,r,save_path=args.save_path+args.save_label+"_patch"+".jpg")
#counstruct his adata
adata_omics2=sc.AnnData(adata.obsm["patch_flattern"], dtype="float64")
adata_omics2.obs=adata.obs.copy()
adata_omics2.obsm=adata.obsm.copy()
adata_omics2.obsm['feat'] = AgaeSMO_v1.pca(adata_omics2, n_comps=50)
adata_omics2.obsm['tensor']=adata_omics2.X




adata_omics1.obsm['feat'] = AgaeSMO_v1.pca(adata_omics1[:, adata_omics1.var['highly_variable']], n_comps=50)
adata_omics1.obsm['tensor']=adata_omics1[:, adata_omics1.var['highly_variable']].X.toarray()
# adata_omics1.obsm['tensor']=AgaeSMO_v1.pca(adata_omics1[:, adata_omics1.var['highly_variable']], n_comps=100)



data = AgaeSMO_v1.construct_neighbor_graph(adata_omics1, adata_omics2,6,6,k=1)
# In[20]:


model = AgaeSMO_v1.Train_AgaeSMO(data, 
                             device=device,
                             learning_rate=0.001,
                             weight_decay=0,
                             epochs=args.epoch,
                             loss_weight=[1,1,1,1],
                             loss_fun="sce")

print(model)
# train model
output = model.train()


# In[21]:

adata = adata_omics1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
adata.obsm['AgaeSMO'] = output['AgaeSMO'].copy()
adata.obsm['alpha'] = output['alpha']


# In[22]:


n_cluster=args.n_cluster
tool = 'mclust' # mclust, leiden, and louvain  
AgaeSMO_v1.clustering(adata,refine_=True, key='AgaeSMO', add_key='AgaeSMO', n_clusters=n_cluster, method="mclust", use_pca=True)# visualization

annotation=pd.read_csv("/public/home/off_liukunpeng/project/7_public_data/Data/osfstorage-archive/slide1_annotation.csv")
adata.obs["GrounTruth"]=list(annotation["annotation"])
indexs=AgaeSMO_v1.supervise_index(adata,"AgaeSMO",'GrounTruth')
ARI=indexs["ARI"]

fig, ax = plt.subplots()
sc.pl.spatial(adata, basis='spatial', color='AgaeSMO',ax=ax ,title='AgaeSMO', s=10, show=False)


fig.savefig(args.save_path+args.save_label+f"_AgaeSMO_ARI_{ARI}"+".jpg")

adata.write(args.save_path+args.save_label+".h5ad")



