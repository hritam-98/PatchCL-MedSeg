import torch
import torch.nn.functional as F
import math

def get_embeddings(model, patch_list, studentBool,batch_size=4):
    embedding_list=[]
    for cls in range(len(patch_list)):
        if patch_list[cls] is not None:
            cls_embedding_list = []
            # print('yo',patch_list[cls].shape[0],math.ceil(patch_list[cls].shape[0]/batch_size))
            for i in range(math.ceil(patch_list[cls].shape[0]/batch_size)):
                batch = patch_list[cls][i*batch_size:min((i+1)*batch_size,patch_list[cls].shape[0]),:,:,:]
                if studentBool is True:
                    batch=batch.to(dev)
                emb = model(batch)
                emb=emb.to('cpu')
                # print('emb',emb.shape)
                emb = F.normalize(emb,p=2,dim=1) # Projecting onto hypersphere of radius 1
                cls_embedding_list.append(emb)
            embedding_list.append(torch.cat(cls_embedding_list,dim=0))
        else:
            embedding_list.append(None)
    return embedding_list