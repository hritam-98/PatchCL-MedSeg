import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import math
import time
from torch.utils.data import DataLoader
from utils.stochastic_approx import StochasticApprox
from utils.model import Network
from utils.datasets import LabData,UnlabData
from utils.queues import Embedding_Queues
from utils.CELOSS import CE_loss
from utils.patch_utils import _get_patches
from utils.aug_utils import batch_augment
from utils.get_embds import get_embeddings
from utils.const_reg import consistency_cost
from utils.plg_loss import PCGJCL
from utils.torch_poly_lr_decay import PolynomialLRDecay









if __name__=="__main__":
	dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	stochastic_approx = StochasticApprox(21,0.5,0.8)

	model = Network()
	teacher_model = Network()

	#Turning off gradients for teacher model
	for param in teacher_model.parameters():
    	param.requires_grad=False
        #Esuring mothe the models have same weight
	teacher_model.load_state_dict(model.state_dict())
	model.contrast=False
	teacher_model.contrast = False

	model = nn.DataParallel(model)
	model = model.to(dev)
	teacher_model = nn.DataParallel(teacher_model)
	teacher_model=teacher_model.to(dev)

	embd_queues = Embedding_Queues(21)

	cross_entropy_loss=CE_loss()
	metrics=[smp.utils.metrics.IoU(threshold=0.5)]

	optimizer_pretrain=torch.optim.Adam(model.parameters(),lr=0.001)
	optimizer_ssl=torch.optim.SGD(model.parameters(),lr=0.007)
	scheduler = PolynomialLRDecay(optim, max_decay_steps=200, end_learning_rate=0.0001, power=2.0)

	contrastive_batch_size = 128


	labeled_dataset = LabData()
	unlabeled_dataset = UnlabData()
	labelled_dataloader = DataLoader(labeled_dataset,batch_size=8,shuffle=True)
	unlabeled_dataloader = DataLoader(unlabeled_dataset,batch_size=8,shuffle=True)

	#CONTRASTIVE PRETRAINING (warm up)
	#torch.autograd.set_detect_anomaly(True)
	for c_epochs in range(100): #100 epochs supervised pre training
	    step=0
	    min_loss = math.inf
	    epoch_loss=0
	    #print('Epoch ',c_epochs)

	    for imgs, masks in labelled_dataloader:

	        t1=time.time()
	        with torch.no_grad():

	            #Send psudo masks & imgs to cpu
	            p_masks=masks
	            imgs = imgs

	            #get classwise patch list
	            patch_list = _get_patches(imgs,p_masks)
	            
	            #stochastic approximation filtering and threshold update
	            #qualified_patch_list = stochastic_approx.update(patch_list)
	            qualified_patch_list = patch_list

	            #make augmentations for teacher model
	            augmented_patch_list = batch_augment(qualified_patch_list,contrastive_batch_size)

	            
	            #convert to tensor
	            aug_tensor_patch_list=[]
	            qualified_tensor_patch_list=[]
	            for i in range(len(augmented_patch_list)):
	                if augmented_patch_list[i] is not None:
	                    aug_tensor_patch_list.append(torch.tensor(augmented_patch_list[i]))
	                    qualified_tensor_patch_list.append(torch.tensor(qualified_patch_list[i]))
	                else:
	                    aug_tensor_patch_list.append(None)
	                    qualified_tensor_patch_list.append(None)
	        

	        #get embeddings of qualified patches through student model
	        model=model.train()
	        model.module.contrast=True
	        student_emb_list = get_embeddings(model,qualified_tensor_patch_list,True)

	        #get embeddings of augmented patches through teacher model
	        teacher_model.train()
	        teacher_model.contrast = True
	        teacher_embedding_list = get_embeddings(teacher_model,aug_tensor_patch_list,False)

	        #enqueue these
	        embd_queues.enqueue(teacher_embedding_list)

	        #calculate PCGJCL loss
	        PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, 128, 0.2 , 4, psi=4096)

	        #calculate supervied loss
	        imgs, masks =imgs.to(dev), masks.to(dev)
	        out = model(imgs)
	        supervised_loss = cross_entropy_loss(out,masks)

	        #total loss
	        loss = supervised_loss + 0.5*PCGJCL_loss

	        epoch_loss+=loss
	        
	        #backpropagate
	        loss.backward()
	        optimizer_contrast.step()


	        for param_stud, param_teach in zip(model.parameters(),teacher_model.parameters()):
	            param_teach.data.copy_(0.001*param_stud + 0.999*param_teach)

	        #Extras
	        t2=time.time()
	        print('step ', step, 'loss: ',loss, ' & time: ',t2-t1)
	        step+=1
	    if epoch_loss < min_loss:
	        torch.save(model,'./best_contrast.pth')


	for c_epochs in range(200): #200 epochs supervised SSL
	    step=0
	    min_loss = math.inf
	    epoch_loss=0
	    #print('Epoch ',c_epochs)

	    labeled_iterator = iter(labelled_dataloader)
	    for imgs in unlabeled_dataloader:

	        t1=time.time()
	        with torch.no_grad():

	            #send imgs to dev
	            imgs = imgs.to(dev)
	            
	            #set model in Eval mode
	            model = model.eval()

	            #Get pseudo masks
	            model.module.contrast=False
	            p_masks = model(imgs)

	            #Send psudo masks & imgs to cpu
	            p_masks=masks
	            p_masks = p_masks.to('cpu').detach()
	            imgs = imgs.to('cpu').detach()

	            #Since we use labeled data for PCGJCL as well
	            imgs2, masks2 = labeled_iterator.next()

	            #concatenating unlabeled and labeled sets
	            p_masks = torch.cat([p_masks,masks2],dim=0)
	            imgs = torch.cat([imgs,imgs2],dim=0)

	            #get classwise patch list
	            patch_list = _get_patches(imgs,p_masks)
	            
	            #stochastic approximation filtering and threshold update
	            qualified_patch_list = stochastic_approx.update(patch_list)


	            #make augmentations for teacher model
	            augmented_patch_list = batch_augment(qualified_patch_list,contrastive_batch_size)

	            #convert to tensor
	            aug_tensor_patch_list=[]
	            qualified_tensor_patch_list=[]
	            for i in range(len(augmented_patch_list)):
	                if augmented_patch_list[i] is not None:
	                    aug_tensor_patch_list.append(torch.tensor(augmented_patch_list[i]))
	                    qualified_tensor_patch_list.append(torch.tensor(qualified_patch_list[i]))
	                else:
	                    aug_tensor_patch_list.append(None)
	                    qualified_tensor_patch_list.append(None)
	        

	        #get embeddings of qualified patches through student model
	        model=model.train()
	        model.module.contrast=True
	        student_emb_list = get_embeddings(model,qualified_tensor_patch_list,True)

	        #get embeddings of augmented patches through teacher model
	        teacher_model.train()
	        teacher_model.contrast = True
	        teacher_embedding_list = get_embeddings(teacher_model,aug_tensor_patch_list,False)

	        #enqueue these
	        embd_queues.enqueue(teacher_embedding_list)

	        #calculate PCGJCL loss
	        PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, 128, 1 , 10, alpha=1)


	        #calculate supervied loss
	        imgs2, masks2 =imgs2.to(dev), masks2.to(dev)
	        out = model(imgs)
	        supervised_loss = cross_entropy_loss(out,masks2)


	        #Consistency Loss
	        consistency_loss=consistency_cost(model,teacher_model,imgs,p_masks)


	        #total loss
	        loss = supervised_loss + 0.5*PCGJCL_loss + 4*consistency_loss
	        
	        #backpropagate
	        loss.backward()
	        optimizer_ssl.step()
	        scheduler.step()


	        for param_stud, param_teach in zip(model.parameters(),teacher_model.parameters()):
	            param_teach.data.copy_(0.001*param_stud + 0.999*param_teach)

	        #Extras
	        t2=time.time()
	        print('step ', step, 'loss: ',loss, ' & time: ',t2-t1)
	        step+=1
	    if epoch_loss < min_loss:
	        torch.save(model,'./best_contrast.pth')
