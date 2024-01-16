"""


Deterministic Policy Action_Comm FC -> GRU -> GAT -> FC


@author: anastasios
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geom_nn
import torch.functional as F
import os

# Define a policy network with communication
class CommPolicyNet(nn.Module):
    def __init__(self, lr, input_dims, n_actions, input_msg_dims, msg_size, fc1_dims, fc2_dims, hidden_size, hidden_channels, num_layers, name, chkpt_dir='tmp/actor'):
        self.lr=lr
        self.input_dims = input_dims # input obs siz
        self.n_actions = n_actions # n_of actions from sim
        self.input_msg_dims = input_msg_dims # input message vector size
        self.msg_size = msg_size # output msg size 
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.hidden_size = hidden_size
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers # num of stacked (GRU), message passing (GAT) layers
        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_act')
	    
	    ## Structure
	    # input encoding layers
        self.fc1 = nn.Linear(*input_dims, fc1_dims) # obs encoding
        self.fc2 = nn.Linear(*msg_size, fc2_dims) # message encoding
	    
	    # temporal dependencies - 2 stacked GRU layers
        self.gru = nn.GRU(fc2_dims, hidden_size, num_layers=2)
	    
	    # spatial dependencies - 2 message passing layers (k=2 hops) - 8 heads each
        self.gat = geom_nn.GAT(hidden_size, hidden_channels, num_layers=2, kwargs={"heads=8"})
	    
	    # communication decision
        self.comm = nn.Linear(hidden_channels, 1) 
	        
	    # outpout encoding layers
        self.mu = nn.Linear(hidden_channels + hidden_size, n_actions) # action
        self.msg = nn.Linear(hidden_channels, msg_size) # message
	    
	    # regularisation - dropout 0.2 probability
        self.dropout1 = nn.Dropout(0.2) # dropout after fc1
        self.dropout2 = nn.Dropout(0.2) # dropout after fc2
        self.dropout3 = nn.Dropout(0.2) # dropout after gru
        self.dropout4 = nn.Dropout(0.2) # dropout after gat
	    
	    # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      	self.optimizer, mode='max', factor=0.1, patience=10)
            
        # Cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:1")
        
        # To device
        self.to(self.device)
                                     	    
    def forward(self, state, message, edge_index):
	    
	    # edge_index is a 2 x E tensor, where E the number of edges,
	    # and each column is a pair of node indices
	    # can create edge_index from adjacency matrix using 		     		
	    # torch_geometric.utils.dense_to_sparse
	    
	    # apply fc1 on observations
        x = self.fc1(state)
        x = F.relu(x)
        x = self.dropout1(x)
	    
	    # apply fc2 on messages
        msg = self.fc2(message)
        msg = F.relu(msg)
        msg = self.dropout2(x)
        
        # reshape msg to match x
        msg = msg.view(-1, 1, self.fc2_dims)
        
        # sum to create a combined representation
        x = x+msg
	    
	    # apply GRU
        x, h = self.gru(x)
        x = F.relu(x)
        x = self.dropout3(x)
	    
	    # apply GAT
        x_gat = self.gat(x, edge_index)
        x_gat = self.dropout4(x_gat)
	    
	    # Comm decision probability
        comm = torch.sigmoid(self.comm(x_gat))
	    
        # Action - concat GRU and GAT as input
        mu = torch.tanh(self.mu(torch.concat(x, x_gat, dims=-1)))
	    # Message
        msg = torch.tanh(self.msg(x_gat))
	    
        return comm, msg, mu
	 
    def save_chpnt(self):
	    print('Saving checkpoint...')
	    torch.save(self.state_dictionary, self.checkpoint_file)
	    
    def load_chkpnt(self):
	    print('Loading checkpoint...')
	    self.load_state_dictionary(torch.load(self.checkpoint_file))
	    
	    
	    
	    

	     
