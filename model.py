import torch
import torch.nn as nn
from evaluate import Engine

class NCF(torch.nn.Module):
  def __init__(self,num_users,num_items,latent_dim_mf,num_layers):
        super(NCF,self).__init__()
        self.user_embedding_gmf = nn.Embedding(num_users,latent_dim_mf)
        self.song_embedding_gmf = nn.Embedding(num_items,latent_dim_mf)
        self.user_embedding_mlp = nn.Embedding(num_users,latent_dim_mf*(2**(num_layers-1)))
        self.song_embedding_mlp = nn.Embedding(num_items,latent_dim_mf*(2**(num_layers-1)))
   
        MLP_modules = []
        for i in range(num_layers):
            input_size = latent_dim_mf*(2**(num_layers-i))
            MLP_modules.append(nn.Linear(input_size,input_size//2))
            MLP_modules.append(nn.ReLU())

        self.MLP_layers =nn.Sequential(*MLP_modules)
        self.predict_layer = nn.Linear(latent_dim_mf*2,1)
        self._init_weight_()
  
  def _init_weight_(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.song_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.song_embedding_mlp.weight, std=0.01)
    
  def forward(self,user_indices,item_indices):
    x1=self.user_embedding_gmf(user_indices)
    x2=self.song_embedding_gmf(item_indices)
    x3=self.user_embedding_mlp(user_indices)
    x4=self.song_embedding_mlp(item_indices)
    
    element_product=torch.mul(x1,x2)
    element_cat=torch.cat((x3,x4),-1)

    output_MLP = self.MLP_layers(element_cat)
    x=torch.cat((element_product,output_MLP),-1)
    x=self.predict_layer(x)
    return x.view(-1)
 
 