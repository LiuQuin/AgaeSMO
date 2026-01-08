import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        
        attention = torch.where(adj> 0, e, zero_vec)
        # print(attention.shape)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        self.attention=attention
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # print(Wh1.shape,Wh2.shape)
        # broadcast add
        e = Wh1 + Wh2.T
        # print(e.shape)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# class SpecialSpmmFunction(torch.autograd.Function):
#     """Special function for only sparse region backpropataion layer."""
#     @staticmethod
#     def forward(ctx, indices, values, shape, b):
#         assert indices.requires_grad == False
#         a = torch.sparse_coo_tensor(indices, values, shape)
#         ctx.save_for_backward(a, b)
#         ctx.N = shape[0]
#         return torch.matmul(a, b)

#     @staticmethod
#     def backward(ctx, grad_output):
#         a, b = ctx.saved_tensors
#         grad_values = grad_b = None
#         if ctx.needs_input_grad[1]:
#             grad_a_dense = grad_output.matmul(b.t())
#             edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
#             grad_values = grad_a_dense.view(-1)[edge_idx]
#         if ctx.needs_input_grad[3]:
#             grad_b = a.t().matmul(grad_output)
#         return None, grad_values, None, grad_b


# class SpecialSpmm(nn.Module):
#     def forward(self, indices, values, shape, b):
#         return SpecialSpmmFunction.apply(indices, values, shape, b)

    
# class GraphAttentionLayer(nn.Module):
#     """
#     Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """

#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat

#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_normal_(self.W.data, gain=1.414)
                
#         self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
#         nn.init.xavier_normal_(self.a.data, gain=1.414)

#         self.dropout = nn.Dropout(dropout)
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         self.special_spmm = SpecialSpmm()

#     def forward(self, input, adj):
#         # dv = 'cuda' if input.is_cuda else 'cpu'
#         dv=input.device
#         N = input.size()[0]
#         edge = adj.nonzero().t()

#         h = torch.mm(input, self.W)
#         # h: N x out
#         assert not torch.isnan(h).any()

#         # Self-attention on the nodes - Shared attention mechanism
#         edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
#         # edge: 2*D x E

#         edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
#         assert not torch.isnan(edge_e).any()
#         # edge_e: E

#         e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
#         # e_rowsum: N x 1

#         edge_e = self.dropout(edge_e)
#         # edge_e: E

#         h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
#         assert not torch.isnan(h_prime).any()
#         # h_prime: N x out
        
#         h_prime = h_prime.div(e_rowsum)
#         # h_prime: N x out
#         assert not torch.isnan(h_prime).any()

#         if self.concat:
#             # if this layer is not last layer,
#             return F.elu(h_prime)
#         else:
#             # if this layer is last layer,
#             return h_prime

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class transfor_graph_attention_layer(nn.Module):

    def __init__(self, 
                 emb1_feat,
                 emb2_feat,
                 out_feat,
                 alpha,dropout,concat=True):
        """n1 emb1_feat"""
        super(transfor_graph_attention_layer, self).__init__()
        self.alpha = alpha
        self.concat = concat

        self.w0_1=nn.Parameter(torch.empty(size=(emb1_feat, out_feat)))
        nn.init.xavier_uniform_(self.w0_1.data, gain=1.414)

        self.w0_2=nn.Parameter(torch.empty(size=(emb2_feat, out_feat)))
        nn.init.xavier_uniform_(self.w0_2.data, gain=1.414)

        self.w1=nn.Parameter(torch.empty(size=(out_feat, 1)))
        nn.init.xavier_uniform_(self.w1.data, gain=1.414)
        
        self.w2=nn.Parameter(torch.empty(size=(out_feat, 1)))
        nn.init.xavier_uniform_(self.w2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.dropout=dropout

    def forward(self, emb1, emb2, adj):
        
        emb1=torch.mm(emb1,self.w0_1)

        emb2=torch.mm(emb2, self.w0_2)

        wh1 = torch.matmul(emb1, self.w1)

        wh2 = torch.matmul(emb2, self.w2)

        e=wh1.T+wh2
        e=self.leakyrelu(e)
        
        zero_vec = -9e15*torch.ones_like(e)
        
        attention = torch.where(adj > 0, e, zero_vec)

        

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        self.attention=attention
        h_1 = torch.matmul(attention, emb1)
        h_2 = torch.matmul(attention.T, emb2)

        if self.concat:
            return F.elu(h_1),F.elu(h_2),attention
        else:
            return h_1,h_2,attention

class AttentionLayer(Module):
    
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.     

    Returns
    -------
    Aggregated representations and modality weights.

    """
    
    def __init__(self, in_feat, out_feat):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined), self.alpha


class encoding_mask_noise(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(encoding_mask_noise, self).__init__()   
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.enc_mask_token = nn.Parameter(torch.zeros(size=(1, in_dim)))
        self.reset_parameters_for_token()
        
    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token.data, gain=1.414)#
        
    def forward(self, x, mask_rate=0.5, replace_rate=0.05):
        # num_nodes = g.num_nodes()
        num_nodes = x.size()[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_token_rate = 1-replace_rate
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]
        
        if replace_rate > 0.0:
            num_noise_nodes = int(replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: -num_noise_nodes]]#int(mask_token_rate * num_mask_nodes)
            noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            # out_x[token_nodes] = torch.zeros_like(out_x[token_nodes])
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
            # out_x[noise_nodes] = torch.add(x[noise_to_be_chosen], out_x[noise_nodes]) 
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        # use_g = g.clone()
        return out_x, mask_nodes, keep_nodes
    
class random_remask(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(random_remask, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.dec_mask_token = nn.Parameter(torch.zeros(size=(1, out_dim)))
        self.reset_parameters_for_token()
        
    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.dec_mask_token.data, gain=1.414)
        
    def forward(self,rep,remask_rate=0.5):
        num_nodes = rep.size()[0]
        # num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes: ]

        out_rep = rep.clone()
        out_rep[remask_nodes] = 0.0
        out_rep[remask_nodes] += self.dec_mask_token
        return out_rep, remask_nodes, rekeep_nodes
