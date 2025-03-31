import torch
import numpy as np
import torch_geometric as tg
import torch_geometric.utils as tgu

# def process_image(image):
#     cord_x = torch.arange(image.shape[-1]).repeat(image.shape[-2], 1)[None]/64
#     cord_y = torch.arange(image.shape[-2]).repeat(image.shape[-1], 1).T[None]/64
#     processed = torch.cat([image, cord_x.to(image.device), cord_y.to(image.device)]).to(torch.float32)
#     return processed.flatten(start_dim=1).T

def process_image(_image):
    image = _image.reshape([4,16,16])
    cord_x = torch.arange(image.shape[-1]).repeat(image.shape[-2], 1)[None]/64
    cord_y = torch.arange(image.shape[-2]).repeat(image.shape[-1], 1).T[None]/64
    processed = torch.cat([image, cord_x.to(image.device), cord_y.to(image.device)]).to(torch.float32)
    return processed.flatten(start_dim=1).T

def build_gnn_batch(**kwargs):
    batch_size = list(kwargs.values())[0].shape[0]
    samples = [tg.data.HeteroData() for _ in range(batch_size)]
    # for arg_name, values in kwargs.items():

    for sample, value in zip(samples, kwargs["x"]):
        node_feats = process_image(value)

        sample["x"].x = node_feats
        #print('%%%%%%%%%%%%%%%% ', node_feats.shape)
        adj = torch.ones(node_feats.shape[0], node_feats.shape[0])
        sample["x", 'to', "x"].edge_index = tgu.dense_to_sparse(adj)

    for arg_name in ['t', 'cat']:
        for sample, value in zip(samples, kwargs[arg_name]):
            sample[arg_name].x = value[None].to(torch.float32)
            adj = torch.ones(1, sample["x"].x.shape[0])
            sample[arg_name, 'to', "x"].edge_index = tgu.dense_to_sparse(adj)
    
    return tg.data.Batch.from_data_list(samples)
    

# def extract_prediction(x_batch):
#     return x_batch.reshape([-1,32,32,1]).permute([0,3,1,2])

def extract_prediction(x_batch):
    #print('$#################### ', x_batch.shape)
    return x_batch.reshape([-1,16,16,4]).permute([0,3,1,2]).reshape([-1,1,32,32])
    