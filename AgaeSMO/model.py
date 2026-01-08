
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .layer import *



class AgaeSMO(Module):
      
     
    def __init__(self, 
                 dim_in_feat_omics1, 
                 dim_out_feat_omics1, 
                 dim_in_feat_omics2, 
                 dim_out_feat_omics2,
                 graph_transfor=None,
                 num_hidden=512):
        super(AgaeSMO, self).__init__()
        
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics2 = dim_out_feat_omics2

        self.num_hidden=num_hidden

        if graph_transfor!=None:
            
            self.graph_transfer=transfor_graph_attention_layer(dim_out_feat_omics1,dim_out_feat_omics2,dim_out_feat_omics1,0.2,0.2)

            self.reverse_transfer=nn.Linear(self.dim_out_feat_omics2,self.dim_out_feat_omics2)
            


        self.encoder_omics1 = GraphAttentionLayer(self.dim_in_feat_omics1, self.num_hidden,0.2,0.2)
        self.encoder_omics1_2 = GraphAttentionLayer(self.num_hidden, self.dim_out_feat_omics1,0.2,0.2)

        self.encoder_omics2 = GraphAttentionLayer(self.dim_in_feat_omics2, self.num_hidden,0.2,0.2)
        self.encoder_omics2_2 = GraphAttentionLayer(self.num_hidden, self.dim_out_feat_omics2,0.2,0.2)


        self.decoder_omics1 = GraphAttentionLayer(self.dim_out_feat_omics1, self.num_hidden,0.2,0.2)
        self.decoder_omics1_2 = GraphAttentionLayer(self.num_hidden, self.dim_in_feat_omics1,0.2,0.2)

        self.decoder_omics2 = GraphAttentionLayer(self.dim_out_feat_omics2, self.num_hidden,0.2,0.2)
        self.decoder_omics2_2 = GraphAttentionLayer(self.num_hidden, self.dim_in_feat_omics2,0.2,0.2)

        

        self.ac=nn.LeakyReLU(0.2)
        
        self.knn_spatial_attention_1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.knn_spatial_attention_2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)

        self.cross_omics_attention = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)

    def forward(self, 
                features_omics1, 
                features_omics2, 

                feat_adj_omics1, 
                spatial_adj_omics1, 
                
                feat_adj_omics2,
                spatial_adj_omics2,
                
                transfer_adj=None
                ):
        
        # graph1
        feat_emb_latent_omics1 = self.encoder_omics1(features_omics1, feat_adj_omics1)
        feat_emb_latent_omics1 = self.encoder_omics1_2(feat_emb_latent_omics1, feat_adj_omics1)

        spatial_emb_latent_omics1 = self.encoder_omics1(features_omics1, spatial_adj_omics1)
        spatial_emb_latent_omics1 = self.encoder_omics1_2(spatial_emb_latent_omics1, spatial_adj_omics1)

        feat_emb_latent_omics2 = self.encoder_omics2(features_omics2, feat_adj_omics2)
        feat_emb_latent_omics2 = self.encoder_omics2_2(feat_emb_latent_omics2, feat_adj_omics2)

        spatial_emb_latent_omics2 = self.encoder_omics2(features_omics2, spatial_adj_omics2)
        spatial_emb_latent_omics2 = self.encoder_omics2_2(spatial_emb_latent_omics2, spatial_adj_omics2)

        emb_latent_omics1,_=self.knn_spatial_attention_1(feat_emb_latent_omics1,spatial_emb_latent_omics1)
        emb_latent_omics2,_=self.knn_spatial_attention_2(feat_emb_latent_omics2,spatial_emb_latent_omics2)


        

        if transfer_adj!=None:
            h_1,emb_latent_omics2,attention=self.graph_transfer(emb_latent_omics1,emb_latent_omics2,transfer_adj.T)
            # emb_latent_omics2=torch.mm(attention.T,emb_latent_omics2)
        
        emb_latent,alpha=self.cross_omics_attention(emb_latent_omics1,emb_latent_omics2)


        recon_ommics1 = self.decoder_omics1(emb_latent, spatial_adj_omics1)
        recon_ommics1 = self.decoder_omics1_2(recon_ommics1, spatial_adj_omics1)

        if transfer_adj==None:
            recon_ommics2 = self.decoder_omics2(emb_latent, spatial_adj_omics2)
            recon_ommics2 = self.decoder_omics2_2(recon_ommics2, spatial_adj_omics2)
        else:
            reverse_graph_transfer= self.reverse_transfer(emb_latent)
            reverse_graph_transfer=self.ac(reverse_graph_transfer)
            reverse_graph_transfer=torch.mm(attention,reverse_graph_transfer)

            recon_ommics2 = self.decoder_omics2(reverse_graph_transfer, spatial_adj_omics2)
            recon_ommics2 = self.decoder_omics2_2(recon_ommics2, spatial_adj_omics2)


        #corr
        corr_feat_emb_latent_omics1 = self.encoder_omics1(recon_ommics1, feat_adj_omics1)
        corr_feat_emb_latent_omics1 = self.encoder_omics1_2(corr_feat_emb_latent_omics1, feat_adj_omics1)

        corr_spatial_emb_latent_omics1 = self.encoder_omics1(features_omics1, spatial_adj_omics1)
        corr_spatial_emb_latent_omics1 = self.encoder_omics1_2(corr_spatial_emb_latent_omics1, spatial_adj_omics1)

        corr_feat_emb_latent_omics2 = self.encoder_omics2(recon_ommics2, feat_adj_omics2)
        corr_feat_emb_latent_omics2 = self.encoder_omics2_2(corr_feat_emb_latent_omics2, feat_adj_omics2)

        corr_spatial_emb_latent_omics2 = self.encoder_omics2(recon_ommics2, spatial_adj_omics2)
        corr_spatial_emb_latent_omics2 = self.encoder_omics2_2(corr_spatial_emb_latent_omics2, spatial_adj_omics2)

        corr_emb_latent_omics1,_=self.knn_spatial_attention_1(corr_feat_emb_latent_omics1,corr_spatial_emb_latent_omics1)
        corr_emb_latent_omics2,_=self.knn_spatial_attention_2(corr_feat_emb_latent_omics2,corr_spatial_emb_latent_omics2)

        if transfer_adj!=None:
            h_1,corr_emb_latent_omics2,_=self.graph_transfer(corr_emb_latent_omics1,corr_emb_latent_omics2,transfer_adj.T)
            # emb_latent_omics2=torch.mm(attention.T,emb_latent_omics2)

        results = {
                   'emb_latent_combined':emb_latent,
                   'spatial_recon_ommics1':recon_ommics1,
                   'spatial_recon_ommics2':recon_ommics2,

                   'corr_emb_latent_omics1':corr_emb_latent_omics1,
                   'corr_emb_latent_omics2':corr_emb_latent_omics2,

                   'emb_latent_omics1':emb_latent_omics1,
                   'emb_latent_omics2':emb_latent_omics2,

                   'alpha':alpha
                   }
        
        return results     




class SSAAA(Module):
      
     
    def __init__(self, 
                 dim_in_feat_omics1, 
                 dim_out_feat_omics1,
                 mask_rate=0.5,
                 replace_rate=0.05,
                 remask_rate=0.5,
                 num_hidden=512):
        super(SSAAA, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_out_feat_omics1 = dim_out_feat_omics1

        self.mask_rate=mask_rate
        self.replace_rate=replace_rate
        self.remask_rate=remask_rate

        self.num_hidden=num_hidden


        # self.mask_1=encoding_mask_noise([dim_in_feat_omics1,num_hidden,dim_out_feat_omics1])
        # self.remask_1=random_remask([dim_in_feat_omics1,num_hidden,dim_out_feat_omics1])



        self.encoder_1 = GraphAttentionLayer(self.dim_in_feat_omics1, self.num_hidden,0.2,0.2)
        self.encoder_2 = GraphAttentionLayer(self.num_hidden, self.dim_out_feat_omics1,0.2,0.2)

        self.decoder_1 = GraphAttentionLayer(self.dim_out_feat_omics1, self.num_hidden,0.2,0.2)
        self.decoder_2 = GraphAttentionLayer(self.num_hidden, self.dim_in_feat_omics1,0.2,0.2)

        self.att_omics1=AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)

    def forward(self, 
                features_omics1, 
                feat_adj_omics1, 
                spatial_adj_omics1,
                ):
        

        #mask
        # features_omics1,self.mask_node_omics_1,self.keep_node_omics_1=self.mask_1(features_omics1,self.mask_rate,self.replace_rate)
#==========================================
        #encoder 1
        feat_emb_latent_omics1 = self.encoder_1(features_omics1, feat_adj_omics1)
        feat_emb_latent_omics1 = self.encoder_2(feat_emb_latent_omics1, feat_adj_omics1)

        spatial_emb_latent_omics1 = self.encoder_1(features_omics1, spatial_adj_omics1)
        spatial_emb_latent_omics1 = self.encoder_2(spatial_emb_latent_omics1, spatial_adj_omics1)


        #attention combine
        emb_latent_omics1,alpha=self.att_omics1(feat_emb_latent_omics1,spatial_emb_latent_omics1)
        
        recon=self.decoder_1(emb_latent_omics1,spatial_adj_omics1)
        recon=self.decoder_2(recon,spatial_adj_omics1)

        corr_feature=self.encoder_1(recon,feat_adj_omics1)
        corr_feature=self.encoder_2(corr_feature,feat_adj_omics1)

        corr_spatial=self.encoder_1(recon,spatial_adj_omics1)
        corr_spatial=self.encoder_2(corr_spatial,spatial_adj_omics1)

        corr,_=self.att_omics1(corr_feature,corr_spatial)

        #remask
        # remasked_emb_latent_combined_1,self.combined_mask_node,self.combined_keep_node=self.remask_1(emb_latent_omics1,remask_rate=self.remask_rate)
        # remasked_emb_latent_combined_2,self.combined_mask_node,self.combined_keep_node=self.remask_1(emb_latent_omics1,remask_rate=self.remask_rate)

#==========================================
        #decoder 1 keep
        # recon_ommics1_keep = self.decoder_1(remasked_emb_latent_combined_1, spatial_adj_omics1)
        # recon_ommics1_keep = self.decoder_2(recon_ommics1_keep, spatial_adj_omics1)
        

        # recon_ommics1_mask = self.decoder_1(remasked_emb_latent_combined_2, spatial_adj_omics1)
        # recon_ommics1_mask = self.decoder_2(recon_ommics1_mask, spatial_adj_omics1)

#==========================================
#corr
        # keep_corr_feat_emb_latent_omics1 = self.encoder_1(recon_ommics1_keep, feat_adj_omics1)
        # keep_corr_feat_emb_latent_omics1 = self.encoder_2(keep_corr_feat_emb_latent_omics1, feat_adj_omics1)

        # keep_corr_spatial_emb_latent_omics1 = self.encoder_1(recon_ommics1_keep, spatial_adj_omics1)
        # keep_corr_spatial_emb_latent_omics1 = self.encoder_2(keep_corr_spatial_emb_latent_omics1, spatial_adj_omics1)

        # keep_corr_emb_latent_omics1,_=self.att_omics1(keep_corr_feat_emb_latent_omics1,keep_corr_spatial_emb_latent_omics1)

        # mask_corr_feat_emb_latent_omics1 = self.encoder_1(recon_ommics1_mask, feat_adj_omics1)
        # mask_corr_feat_emb_latent_omics1 = self.encoder_2(mask_corr_feat_emb_latent_omics1, feat_adj_omics1)

        # mask_corr_spatial_emb_latent_omics1 = self.encoder_1(recon_ommics1_mask, spatial_adj_omics1)
        # mask_corr_spatial_emb_latent_omics1 = self.encoder_2(mask_corr_spatial_emb_latent_omics1, spatial_adj_omics1)

        # mask_corr_emb_latent_omics1,_=self.att_omics1(mask_corr_feat_emb_latent_omics1,mask_corr_spatial_emb_latent_omics1)

        results = {
            'emb_latent_omics1':emb_latent_omics1,
            # "keep_corr_emb_latent_omics1":keep_corr_emb_latent_omics1,
            # 'mask_corr_emb_latent_omics1':mask_corr_emb_latent_omics1,
            # "recon_ommics1_keep":recon_ommics1_keep,
            # "recon_ommics1_mask":recon_ommics1_mask,
            # "emb_latent_omics1":emb_latent_omics1,
            "recon":recon,
            "corr":corr,
            'alpha':alpha,
                   }
        
        return results
    




# class SSAAA(Module):
      
     
#     def __init__(self, 
#                  dim_in_feat_omics1, 
#                  dim_out_feat_omics1):
#         super(SSAAA, self).__init__()
#         self.dim_in_feat_omics1 = dim_in_feat_omics1
#         self.dim_out_feat_omics1 = dim_out_feat_omics1
#         self.num_hidden=512

#         # self.feat_encoder_omics1 = GraphAttentionLayer(self.dim_in_feat_omics1, self.num_hidden,0.2,0.2)

#         # self.feat_encoder_omics2 = GraphAttentionLayer(self.num_hidden, self.dim_out_feat_omics1,0.2,0.2)
                
#         self.spatial_encoder_omics1 = GraphAttentionLayer(self.dim_in_feat_omics1, self.num_hidden,0.2,0.2)

#         self.spatial_encoder_omics2 = GraphAttentionLayer(self.num_hidden, self.dim_out_feat_omics1,0.2,0.2)

#         # self.att_omics1=AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)

#         # self.feat_decoder_omics1 = GraphAttentionLayer(self.dim_out_feat_omics1, self.num_hidden,0.2,0.2)

#         # self.feat_decoder_omics2 = GraphAttentionLayer(self.num_hidden, self.dim_in_feat_omics1,0.2,0.2)
        
#         self.spatial_decoder_omics1 = GraphAttentionLayer(self.dim_out_feat_omics1, self.num_hidden,0.2,0.2)

#         self.spatial_decoder_omics2 = GraphAttentionLayer(self.num_hidden, self.dim_in_feat_omics1,0.2,0.2)

#     def forward(self, 
#                 features_omics1, 
#                 feat_adj_omics1, 
#                 spatial_adj_omics1,
#                 ):
        
#         #omics_1_feat_spatial
#         # feat_emb_latent_omics1 = self.feat_encoder_omics1(features_omics1, feat_adj_omics1)

#         # feat_emb_latent_omics1 = self.feat_encoder_omics2(feat_emb_latent_omics1, feat_adj_omics1)

#         spatial_emb_latent_omics1 = self.spatial_encoder_omics1(features_omics1, spatial_adj_omics1)

#         emb_latent_omics1 = self.spatial_encoder_omics2(spatial_emb_latent_omics1, spatial_adj_omics1)

#         # emb_latent_omics1,alpha_omics1=self.att_omics1(feat_emb_latent_omics1,spatial_emb_latent_omics1)
        
#         # feat_recon_ommics1 = self.feat_decoder_omics1(emb_latent_omics1, spatial_adj_omics1)

#         # feat_recon_ommics1 = self.feat_decoder_omics2(feat_recon_ommics1, spatial_adj_omics1)
        
#         spatial_recon_ommics1 = self.spatial_decoder_omics1(emb_latent_omics1, spatial_adj_omics1)

#         spatial_recon_ommics1 = self.spatial_decoder_omics2(spatial_recon_ommics1, spatial_adj_omics1)


#         results = {'emb_latent_omics1':emb_latent_omics1,
#                 #    'feat_recon_ommics1':feat_recon_ommics1,
#                    'spatial_recon_ommics1':spatial_recon_ommics1,
#                     # 'feat_emb_latent_omics1':feat_emb_latent_omics1,
#                     # 'spatial_emb_latent_omics1':spatial_emb_latent_omics1,
#                 #    'alpha':alpha_omics1,
#                    }
        
#         return results    