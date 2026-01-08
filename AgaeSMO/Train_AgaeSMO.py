import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import *
from .utils import *


class Train_AgaeSMO:
    def __init__(self, 
        data,
        device= torch.device('cuda'),
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=30,
        trans_ngb=6,
        loss_weight=[1,1,1,1],
        loss_fun='sce'
        ):

        self.data = data.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.loss_weight=loss_weight
        if loss_fun=="sce":
            self.loss_fun=sce_loss
        elif loss_fun=="mse":
            self.loss_fun=nn.MSELoss()
        else:
            None
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        
        if self.adata_omics1.X.shape[0]!=self.adata_omics2.X.shape[0]:
            print("spot number is not equal,adding graph transfer")
            self.adj_o_1_2=construct_adj(self.adata_omics1.obsm["HE_domain_coor"],self.adata_omics2.obsm["HE_domain_coor"],n_neighbors=trans_ngb)
            self.adj_o_1_2=torch.FloatTensor(self.adj_o_1_2).to(self.device)
        else:
            self.adj_o_1_2=None


        # spatial_adj_omics1=transform_adjacent_matrix(self.adata_omics1.uns['adj_spatial']).toarray()
        # spatial_adj_omics2=transform_adjacent_matrix(self.adata_omics2.uns['adj_spatial']).toarray()
        spatial_adj_omics1=self.adata_omics1.uns['adj_spatial'].toarray()
        spatial_adj_omics2=self.adata_omics2.uns['adj_spatial'].toarray()

        self.feat_adj_omics1=torch.FloatTensor(self.adata_omics1.obsm['adj_feature'].toarray()).to(self.device)
        self.spatial_adj_omics1=torch.FloatTensor(spatial_adj_omics1).to(self.device)
        self.feat_adj_omics2=torch.FloatTensor(self.adata_omics2.obsm['adj_feature'].toarray()).to(self.device)
        self.spatial_adj_omics2=torch.FloatTensor(spatial_adj_omics2).to(self.device)
        

        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['tensor'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['tensor'].copy()).to(self.device)
        
        print(self.features_omics1.shape,
              self.features_omics2.shape)

        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        

    
    def train(self):
        print("dim_input1",self.dim_input1,"dim_input2",self.dim_input2,"\n",
              "dim_output1",self.dim_output1,"dim_output2",self.dim_output2)
        self.model = AgaeSMO(self.dim_input1, 
                           self.dim_output1, 
                           self.dim_input2, 
                           self.dim_output2,
                           self.adj_o_1_2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            results = self.model(self.features_omics1, 
                                 self.features_omics2,

                                 self.feat_adj_omics1, 
                                 self.spatial_adj_omics1,
                                 
                                 self.feat_adj_omics2,
                                 self.spatial_adj_omics2,
                                 
                                 self.adj_o_1_2)
            
            # reconstruction loss

            self.loss_spatial_recon_omics1 = self.loss_fun(self.features_omics1, results['spatial_recon_ommics1'])
            self.loss_spatial_recon_omics2 = self.loss_fun(self.features_omics2, results['spatial_recon_ommics2'])
            self.loss_corr_recon_omics1 =    self.loss_fun(results['emb_latent_omics1'], results['corr_emb_latent_omics1'])
            self.loss_corr_recon_omics2 =    self.loss_fun(results['emb_latent_omics2'], results['corr_emb_latent_omics2'])
                
            loss = self.loss_weight[0]*self.loss_spatial_recon_omics1 + \
                self.loss_weight[1]*self.loss_spatial_recon_omics2 + \
                self.loss_weight[2]*self.loss_corr_recon_omics1 + \
                self.loss_weight[3]*self.loss_corr_recon_omics2
            if epoch%10==0:
                print(f"epoch:{epoch}",loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        print("Model training finished!\n")    
    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, 
                               self.features_omics2, 
                               
                               self.feat_adj_omics1, 
                               self.spatial_adj_omics1,
                               
                               self.feat_adj_omics2,
                               self.spatial_adj_omics2,
                               
                               self.adj_o_1_2)
 
        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'AgaeSMO': emb_combined.detach().cpu().numpy(),
                  'alpha': results['alpha'].detach().cpu().numpy()}
        
        return output


class Train_SSAAA:
    def __init__(self, 
        data,
        device= torch.device('cuda'),
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=30,
        mask_rate=0.5,
        replace_rate=0.05,
        remask_rate=0.5,
        loss_weight=[1,1]
        ):

        self.data = data.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.mask_rate=mask_rate
        self.replace_rate=replace_rate
        self.remask_rate=remask_rate

        self.loss_weight=loss_weight
        
        # adj
        self.adata_omics1 = self.data

        spatial_adj_omics1=self.adata_omics1.uns['adj_spatial'].toarray()

        self.feat_adj_omics1=torch.FloatTensor(self.adata_omics1.obsm['adj_feature'].toarray()).to(self.device)
        self.spatial_adj_omics1=torch.FloatTensor(spatial_adj_omics1).to(self.device)

        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['tensor'].copy()).to(self.device)

        print(self.features_omics1.shape)

        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        
        self.dim_output1 = self.dim_output
    
    def train(self):
        print(self.dim_input1,self.dim_output1)
        self.model = SSAAA(self.dim_input1, 
                           self.dim_output1,
                           self.mask_rate,
                           self.replace_rate,
                           self.remask_rate
                           ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
 
            results = self.model(self.features_omics1, 
                                 self.feat_adj_omics1, 
                                 self.spatial_adj_omics1)
            
            # reconstruction loss
            # self.loss_recon_omics1_keep = sce_loss(self.features_omics1[self.model.keep_node_omics_1,:], results['recon_ommics1_keep'][self.model.keep_node_omics_1,:])
            # self.loss_recon_omics1_mask = sce_loss(self.features_omics1[self.model.mask_node_omics_1,:], results['recon_ommics1_mask'][self.model.mask_node_omics_1,:])
            # self.loss_corr_keep = sce_loss(results["emb_latent_omics1"], results['keep_corr_emb_latent_omics1'])
            # self.loss_corr_mask = sce_loss(results["emb_latent_omics1"], results['mask_corr_emb_latent_omics1'])
            # loss= \
            #     self.loss_weight[0]*self.loss_recon_omics1_keep+ \
            #     self.loss_weight[1]*self.loss_recon_omics1_mask+ \
            #     self.loss_weight[2]*self.loss_corr_keep+ \
            #     self.loss_weight[3]*self.loss_corr_mask
            self.loss_recon_omics1=sce_loss(self.features_omics1, results['recon'])
            self.loss_corr_omics1=sce_loss(results['corr'], results['emb_latent_omics1'])
            # loss = self.loss_weight[0]*self.loss_feat_recon_omics1 +  self.loss_weight[1]*self.loss_spatial_recon_omics1
            loss = self.loss_weight[0]*self.loss_recon_omics1+self.loss_weight[1]*self.loss_corr_omics1

            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Model training finished!\n")    
    
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, 
                               self.feat_adj_omics1, 
                               self.spatial_adj_omics1)
 
        # feat_emb_latent_omics1 = F.normalize(results['feat_emb_latent_omics1'], p=2, eps=1e-12, dim=1)  
        # spatial_emb_latent_omics1 = F.normalize(results['spatial_emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        # emb_combined = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)

        # feat_emb_latent_omics1 = results['feat_emb_latent_omics1']
        # spatial_emb_latent_omics1 = results['spatial_emb_latent_omics1']
        emb_combined = results['emb_latent_omics1']
        
        output = {
                # 'feat_emb_latent_omics1': feat_emb_latent_omics1.detach().cpu().numpy(),
                # 'spatial_emb_latent_omics1': spatial_emb_latent_omics1.detach().cpu().numpy(),
                'SSAAA': emb_combined.detach().cpu().numpy(),
                'alpha': results['alpha'].detach().cpu().numpy()
                  }
        
        return output
