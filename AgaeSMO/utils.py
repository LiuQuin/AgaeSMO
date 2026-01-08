import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
import os
import torch
import pandas as pd
import scanpy as sc
import numpy as np 
import torch
import torch.nn.functional as F


def construct_neighbor_graph(adata_omics1, adata_omics2, cutoff1=6,cutoff2=6,k=20): 
    """
    Construct neighbor graphs, including feature graph and spatial graph. 
    Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    data : dict
        AnnData objects with preprossed data for different omics.

    """
        
    Cal_Spatial_Net(adata_omics1,"adj_spatial_", model='KNN',k_cutoff=cutoff1)
    Stats_Spatial_Net(adata_omics1)

    Cal_Spatial_Net(adata_omics2,"adj_spatial_", model='KNN',k_cutoff=cutoff2)
    Stats_Spatial_Net(adata_omics2)
    
    feature_graph_omics1 = construct_graph_by_feature(adata_omics1,k=k)
    feature_graph_omics2 = construct_graph_by_feature(adata_omics2,k=k)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2
    

    adata_omics1.uns["adj_spatial"]=Transfer_adj(adata_omics1,"adj_spatial_")
    adata_omics2.uns["adj_spatial"]=Transfer_adj(adata_omics2,"adj_spatial_")

    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}
    return data

def construct_neighbor_graph_single(adata_omics1, rad_cutoff1=6,k=20): 

    Cal_Spatial_Net(adata_omics1,"adj_spatial_",model='KNN',k_cutoff=rad_cutoff1)
    Stats_Spatial_Net(adata_omics1)

    
    feature_graph_omics1 = construct_graph_by_feature(adata_omics1,k=k)
    adata_omics1.obsm['adj_feature']=feature_graph_omics1
    adata_omics1.uns["adj_spatial"]=Transfer_adj(adata_omics1,"adj_spatial_")

    return adata_omics1


def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     

def construct_graph_by_feature_leiden(adata, resolution=None):
    
    """Constructing feature neighbor graph according to expresss profiles"""
    if "leiden" not in adata.obs:
        sc.tl.leiden(adata)
    mtx=np.eye(adata.obs.shape[0])

    return mtx

def construct_graph_by_feature(adata, k=20, mode= "connectivity", metric="correlation", include_self=False):
    
    """Constructing feature neighbor graph according to expresss profiles"""
    
    feature_graph_omics1=kneighbors_graph(adata.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)

    return feature_graph_omics1

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    #print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)  
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj

def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix"""
    
    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)
    
    adj_spatial_omics1 = adj_spatial_omics1.toarray()   # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()
    
    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1>1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2>1, 1, adj_spatial_omics2)
    
    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1) # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)
    
    ######################################## construct feature graph ########################################
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())
    
    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1>1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2>1, 1, adj_feature_omics2)
    
    # convert dense matrix to sparse matrix
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1) # sparse adjacent matrix corresponding to feature graph
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)
    
    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           'adj_feature_omics1': adj_feature_omics1,
           'adj_feature_omics2': adj_feature_omics2,
           }
    
    return adj

def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   
    
def fix_seed(seed):
    #seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'    


import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
# from .preprocess import pca
import matplotlib.pyplot as plt

#os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'    

# def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
#     """\
#     Clustering using the mclust algorithm.
#     The parameters are the same as those in the R package mclust.
#     """
    
#     np.random.seed(random_seed)
#     import rpy2.robjects as robjects
#     robjects.r.library("mclust")

#     import rpy2.robjects.numpy2ri
#     rpy2.robjects.numpy2ri.activate()
#     r_random_seed = robjects.r['set.seed']
#     r_random_seed(random_seed)
#     rmclust = robjects.r['Mclust']
    
#     res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
#     mclust_res = np.array(res[-2])
    
#     adata.obs['mclust'] = mclust_res
#     adata.obs['mclust'] = adata.obs['mclust'].astype('int')
#     adata.obs['mclust'] = adata.obs['mclust'].astype('category')
#     return adata

def clustering_(adata, n_clusters=7, key='emb', add_key='SMAAA', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']


def clustering(adata, refine_=True,n_clusters=7, key='emb',  method='mclust',add_key='SMAAA',use_pca=False, n_comps=20):
    
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    if use_pca: 
        adata = mclust_R(adata,refine_=refine_, used_obsm=key + '_pca', num_cluster=n_clusters)
    else:
        adata = mclust_R(adata,refine_=refine_, used_obsm=key, num_cluster=n_clusters)
    adata.obs[add_key] = adata.obs['mclust']

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     

def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein',show=True):
  """\
  Plotting weight values
  
  """  
  import pandas as pd  
  
  df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1)
  ax.set_title(modality1 + ' vs ' + modality2) 

  plt.tight_layout(w_pad=0.05)
  if show:
    plt.show()     

def sce_loss(x, y, alpha=3.0):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def combine_adj(adata):
    adj_feat=adata.obsm['adj_feature'].toarray()
    adj_spatial=transform_adjacent_matrix(adata.uns['adj_spatial']).toarray()
    combine=adj_feat+adj_spatial
    combine=combine!=0
    return combine.astype(np.float32)

def construct_neighbor_graph_single_omics(adata_omics1):
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=6)
    adata_omics1.uns['adj_spatial'] = adj_omics1
    feature_graph_omics1=kneighbors_graph(adata_omics1.obsm['feat'], 20, mode= "connectivity", metric="correlation", include_self=False)
    adata_omics1.obsm['adj_feature']=feature_graph_omics1
    return {'adata_omics1': adata_omics1}


#####
def construct_graph_by_coordinate_(st_point,ms_point, n_neighbors):
    #initial st_ppot
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(st_point) 
    #find ms adj point in st field
    distances , indices = nbrs.kneighbors(ms_point)
    return distances,indices

def construct_distance_matrix(st_point,ms_point,n_neighbors=9):
    distances,indices=construct_graph_by_coordinate_(st_point,ms_point,n_neighbors=n_neighbors)
    #init adj matrix
    adj=np.zeros(shape=(len(st_point),len(ms_point)))
    #fill distance matrix
    for i,idx in enumerate(indices):
        adj[idx,i]=distances[i]
    
    # spr_matrix = csr_matrix(adj)
    # spr_matrix.data=1/spr_matrix.data
    return adj

def construct_adj(st_point,ms_point,n_neighbors=9):
    distances,indices=construct_graph_by_coordinate_(st_point,ms_point,n_neighbors=n_neighbors)
    #init adj matrix
    adj=np.zeros(shape=(len(st_point),len(ms_point)))
    #fill adj matrix
    for i,idx in enumerate(indices):
        adj[idx,i]=1
    
    # spr_matrix = csr_matrix(adj)
    # spr_matrix.data=1/spr_matrix.data
    return adj

def construct_spatial_adj(adata_omics1,adata_omics2,n_neighbors1,n_neighbors2):
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors1)
    adata_omics1.uns['adj_spatial'] = adj_omics1

    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors2)
    adata_omics2.uns['adj_spatial'] = adj_omics2

    data={'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}

    return data

def SVG(adata,fold=1):
    gene_name=pd.DataFrame(adata.uns["rank_genes_groups"]['names'])
    pvals_adj=pd.DataFrame(adata.uns["rank_genes_groups"]['pvals_adj'])
    lgfc=pd.DataFrame(adata.uns["rank_genes_groups"]['logfoldchanges'])
    svg=[]
    for i in gene_name.columns:
        cluster_gene=gene_name.loc[
            (pvals_adj.loc[:,i]<0.05) & (lgfc.loc[:,i]>fold),i]
        svg+=list(cluster_gene)
    return list(set(svg))

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




def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['adj_spatial_']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['adj_spatial_']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)

def mclust_R(adata, num_cluster,refine_=True, modelNames='EEE', used_obsm='SMAAA', random_seed=52, dist=None):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    if refine_:
        refined_pred=refine(adata)
        adata.obs["mclust"]=refined_pred
        adata.obs["mclust"]=adata.obs["mclust"].astype('category')
    return adata

def refine(adata=None,obs_key="mclust"):
    refined_pred=[]
    dis_df=adata.uns['adj_spatial_'].reset_index(drop=True)
    sample_id=adata.obs.index.tolist()
    pred=adata.obs[obs_key].tolist()
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    for index in sample_id:
        num_index=dis_df[dis_df.loc[:,'Cell1']==index].index
        num_nbs=len(num_index)
        self_pred=pred.loc[index, "pred"]
        if num_nbs>0:
            dis_tmp=dis_df.loc[num_index,:]
            nbs_pred=pred.loc[dis_tmp.loc[:,'Cell2'].to_list(), "pred"]
           
            v_c=nbs_pred.value_counts()
            if self_pred in v_c.index:
                if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
                    refined_pred.append(v_c.idxmax())
                else:           
                    refined_pred.append(self_pred)
            else:
                refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred



def Cal_Spatial_Net(adata,add_key=None,rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True,delta_err=1):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        distance_threshold=np.sort(distances[:,-1])[0]
        distance_threshold = distance_threshold+delta_err
        for it in range(indices.shape[0]):
            close_indices = indices[it, distances[it, :] <= distance_threshold]
            close_distances = distances[it, distances[it, :] <= distance_threshold]
            KNN_list.append(pd.DataFrame(zip([it]*len(close_indices), close_indices, close_distances)))
            # KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns[add_key] = Spatial_Net

def Transfer_adj(adata,uns_keys):
    G_df = adata.uns[uns_keys].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    return G

def plot_spot_his(img_numpy,spot):
    plt.imshow(img_numpy)
    plt.scatter(spot[:,0],spot[:,1],s=1,c="b")

def check_patch_his(adata,his_dir,r,save_path=None):
    img=adata.uns["spatial"][his_dir]["images"]["hires"]
    fig,ax=plt.subplots(2,5,figsize=(10,4))
    for i in range(5):
        ax[0,i].imshow(img)
        ax[0,i].scatter(adata.obsm["spatial_px"][:,0],adata.obsm["spatial_px"][:,1],s=10,c="b")
        ax[0,i].scatter(adata.obsm["spatial_xp_yp"][:,0],adata.obsm["spatial_xp_yp"][:,1],s=10,c="r")
        ax[0,i].scatter(adata.obsm["spatial_xp_yn"][:,0],adata.obsm["spatial_xp_yn"][:,1],s=10,c="g")
        ax[0,i].scatter(adata.obsm["spatial_xn_yp"][:,0],adata.obsm["spatial_xn_yp"][:,1],s=10,c="y")
        ax[0,i].scatter(adata.obsm["spatial_xn_yn"][:,0],adata.obsm["spatial_xn_yn"][:,1],s=10,c="black")
        ax[0,i].set_xlim(adata.obsm["spatial_px"][i,0]-r*3,adata.obsm["spatial_px"][i,0]+r*3)
        ax[0,i].set_ylim(adata.obsm["spatial_px"][i,1]-r*3,adata.obsm["spatial_px"][i,1]+r*3)

        
        ax[1,i].imshow(adata.obsm["patch"][i])
        ax[1,i].invert_yaxis()
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90, wspace=0.1, hspace=0.1)
    if save_path!=None:
        fig.savefig(save_path,dpi=600)

def calculate_box(adata,xy,r):
    xy_p_x=xy[:,0]+r
    xy_p_y=xy[:,1]+r
    xy_n_x=xy[:,0]-r
    xy_n_y=xy[:,1]-r
    xp_yp=np.vstack([xy_p_x,xy_p_y]).T
    xp_yn=np.vstack([xy_p_x,xy_n_y]).T
    xn_yp=np.vstack([xy_n_x,xy_p_y]).T
    xn_yn=np.vstack([xy_n_x,xy_n_y]).T
    adata.obsm["spatial_xp_yp"]=xp_yp
    adata.obsm["spatial_xp_yn"]=xp_yn
    adata.obsm["spatial_xn_yp"]=xn_yp
    adata.obsm["spatial_xn_yn"]=xn_yn
    return adata

def his_path(adata,img_np):
    img_np
    patch_list=[]
    for i in range(adata.obsm["spatial_px"].shape[0]):
        
        x_l=adata.obsm["spatial_xn_yn"][i,0]
        x_r=adata.obsm["spatial_xp_yp"][i,0]
        y_u=adata.obsm["spatial_xp_yp"][i,1]
        y_d=adata.obsm["spatial_xn_yn"][i,1]
        
        if x_l>0 and x_r<img_np.shape[1] and y_d>0 and y_u<img_np.shape[0]:
            patch_list.append(img_np[y_d:y_u,x_l:x_r,:])
        # else:
        #     x_l=max(adata.obsm["spatial_xn_yn"][i,0],0)
        #     x_r=min(adata.obsm["spatial_xp_yp"][i,0],img_np.shape[0])
        #     y_u=min(adata.obsm["spatial_xp_yp"][i,1],img_np.shape[1])
        #     y_d=max(adata.obsm["spatial_xn_yn"][i,1],0)
            
        #     org_img=img_np[y_d:y_u,x_l:x_r,:]
            
        #     patchs=rand_pad(org_img)
    patch_=np.stack(patch_list)
        # patch_list.append(img_np[x_l:x_r,y_d:y_u,:])
    return patch_
