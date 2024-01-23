
# coding: utf-8

# In[1]:

import os
import math

import tensorflow as tf
#config=tf.ConfigProto()
# add / remove the allow_growth to run the code without errors
#config.gpu_options.allow_growth=True
#config.allow_soft_placement=True

#cuda illegal error access fix
#Error polling for event status: failed to query event: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
#os.environ['TF_CUDNN_DETERMINISTIC']='1'

import matplotlib
matplotlib.use('Agg')

import numpy as np
#to make directories
import pathlib

import sys
sys.path.append('../')

from utils import *

from skimage import transform
import random
random.seed(1)

import time 

import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc','prostate_md','mmwhs'])
#no of training images
parser.add_argument('--no_of_tr_imgs', type=str, default='tr2')
#combination of training images
parser.add_argument('--comb_tr_imgs', type=str, default='c1')
#learning rate of network
#parser.add_argument('--lr_cont_loss', type=float, default=0.001)

# data aug - 0 - disabled, 1 - enabled
parser.add_argument('--data_aug', type=int, default=0, choices=[0,1])
# version of run
parser.add_argument('--ver', type=int, default=0)
# temperature_scaling factor
parser.add_argument('--temp_fac', type=float, default=0.1)
# bounding box dim - dimension of the cropped image. Ex. if bbox_dim=100, then 100 x 100 region is randomly cropped from original image of size W x W & then re-sized to W x W.
# Later, these re-sized images are used for pre-training using global contrastive loss.
parser.add_argument('--bbox_dim', type=int, default=150)

# Config of baseline model
#no of training images - same as no_of_tr_imgs
# parser.add_argument('--pretr_no_of_tr_imgs', type=str, default='tr1')#, choices=['tr52','tr22','tr10'])
# #combination of training images - same as comb_tr_imgs
# parser.add_argument('--pretr_comb_tr_imgs', type=str, default='c1')#, choices=['c1'])
# #no of iterations to run
parser.add_argument('--pretr_n_iter', type=int, default=5001)
# #pretr version - same as ver
# parser.add_argument('--pretr_ver', type=int, default=0)


# type of local_loss_exp_no for Local contrastive loss
# 0 - intra - pixel embedding match - matching within same image for same class
# 1 - inter - pixel embedding match - matching across different images for same class
parser.add_argument('--local_loss_exp_no', type=int, default=1)
#lamda_cont to balance local contrastive loss wrt segmentation loss
parser.add_argument('--lamda_local', type=float, default=0.1)
#no of positive and negative pixel embeddings to sample from an image per class to match to its mean embedding
parser.add_argument('--no_of_neg_eles', type=int, default=3)
parser.add_argument('--no_of_pos_eles', type=int, default=3)

#batch_size value for Local loss computation batch
parser.add_argument('--bt_size', type=int,default=10)
#no of unlabeled images - select it based on dataset name
#parser.add_argument('--no_of_unl_imgs', type=str, default='tr52')
#combination of unlabeled images
#parser.add_argument('--comb_unl_imgs', type=str, default='c1')

#no of iterations to run
parser.add_argument('--n_iter', type=int, default=15001)

#dsc_loss
parser.add_argument('--dsc_loss', type=int, default=2)
#random deformations - arguments
#enable random deformations
parser.add_argument('--rd_en', type=int, default=1)
#sigma of gaussian distribution used to sample random deformations 3x3 grid values
parser.add_argument('--sigma', type=float, default=5)
#enable random contrasts
parser.add_argument('--ri_en', type=int, default=1)
#enable 1-hot encoding of the labels 
parser.add_argument('--en_1hot', type=int, default=1)
#controls the ratio of deformed images to normal images used in each mini-batch of the training
parser.add_argument('--rd_ni', type=int, default=1)
#learning rate of network
parser.add_argument('--lr_seg', type=float, default=0.001)

# Control quality of pseudo-labels used for the joint training
#to enable the measurement of overlap between predicted masks for two augmented versions of an unlabeled volume
parser.add_argument('--test_aug_overlap', type=int, default=0)
#threshold f1_value to select a given unlabeled volume / not
parser.add_argument('--test_f1_threshold', type=float, default=0.9)

parse_config = parser.parse_args()
#parse_config = parser.parse_args(args=[])

# In[2]:

if parse_config.dataset == 'acdc':
    print('load acdc configs')
    import experiment_init.init_acdc as cfg
    import experiment_init.data_cfg_acdc as data_list
    no_of_unl_imgs='tr52'
    comb_unl_imgs ='c1'
elif parse_config.dataset == 'mmwhs':
    print('load mmwhs configs')
    import experiment_init.init_mmwhs as cfg
    import experiment_init.data_cfg_mmwhs as data_list
    no_of_unl_imgs = 'tr10'
    comb_unl_imgs = 'c1'
elif parse_config.dataset == 'prostate_md':
    print('load prostate_md configs')
    import experiment_init.init_prostate_md as cfg
    import experiment_init.data_cfg_prostate_md as data_list
    no_of_unl_imgs = 'tr22'
    comb_unl_imgs = 'c1'
else:
    raise ValueError(parse_config.dataset)

######################################
# class loaders
# ####################################
#  load dataloader object
from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)

if parse_config.dataset == 'acdc':
    print('set acdc orig img dataloader handle')
    orig_img_dt=dt.load_acdc_imgs
elif parse_config.dataset == 'mmwhs':
    print('set mmwhs orig img dataloader handle')
    orig_img_dt=dt.load_mmwhs_imgs
elif parse_config.dataset == 'prostate_md':
    print('set prostate_md orig img dataloader handle')
    orig_img_dt=dt.load_prostate_imgs_md

#  load model object
if(parse_config.ver==1):
    from models_stop_grad import modelObj
    model = modelObj(cfg)
else:
    from models import modelObj
    model = modelObj(cfg)
#  load f1_utils object
from f1_utils import f1_utilsObj
f1_util = f1_utilsObj(cfg,dt)

if(parse_config.rd_en==1):
    parse_config.en_1hot=1
else:
    parse_config.en_1hot=0

######################################
if parse_config.dataset == 'acdc':
    #for acdc
    para_list_fin=[0.01,0.5,10,10,30,30]
elif parse_config.dataset == 'prostate_md':
    #for prostate_md
    para_list_fin=[0.01,0.5,10,5,30,30]
elif parse_config.dataset == 'mmwhs':
    #for mmwhs
    para_list_fin=[0.01,0.5,10,10,30,30]

######################################
# Infer pseudo-labels of unlabeled images from baseline model trained on labeled set
######################################

#directory of saved fine-tuned model on labeled set
baseline_model_dir = str(cfg.srt_dir) + '/models/' + str(parse_config.dataset) + '/trained_models/train_baseline/'

baseline_model_dir = str(baseline_model_dir) + '/with_data_aug/'

if (parse_config.rd_en == 1 and parse_config.ri_en == 1):
    baseline_model_dir = str(baseline_model_dir) + 'rand_deforms_and_ints_en/'
elif (parse_config.rd_en == 1):
    baseline_model_dir = str(baseline_model_dir) + 'rand_deforms_en/'
elif (parse_config.ri_en == 1):
    baseline_model_dir = str(baseline_model_dir) + 'rand_ints_en/'

#baseline_model_dir = str(baseline_model_dir) + str(parse_config.pretr_no_of_tr_imgs) + '/' + str(parse_config.pretr_comb_tr_imgs) + \
#                '_v' + str(parse_config.pretr_ver) + '/unet_dsc_loss_' + str(parse_config.dsc_loss) + \
#                '_n_iter_' + str(parse_config.pretr_n_iter) + '_lr_seg_' + str(parse_config.lr_seg) + '/'
baseline_model_dir = str(baseline_model_dir) + str(parse_config.no_of_tr_imgs) + '/' + str(parse_config.comb_tr_imgs) + \
                '_v' + str(parse_config.ver) + '/unet_dsc_loss_' + str(parse_config.dsc_loss) + \
                '_n_iter_' + str(parse_config.pretr_n_iter) + '_learning_rate_' + str(parse_config.lr_seg) + '/'
print('baseline_model_dir',baseline_model_dir)
######################################


# In[3]:

######################################
tf.reset_default_graph()
# Segmentation Network
ae_ft = model.seg_unet(learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,en_1hot=parse_config.en_1hot,mtask_en=0)

######################################
# Restore the model and initialize the encoder with the pre-trained weights from last epoch
mp_best=get_max_chkpt_file(baseline_model_dir)
print('load best checkpoint model')
print('mp_best',mp_best)

saver_ftnet = tf.train.Saver()
#sess_ftnet = tf.Session(config=config)
sess_ftnet = tf.Session()
saver_ftnet.restore(sess_ftnet, mp_best)
print("Model restored")

######################################
# get all weights- get all variable names and their values
print('Loading trainable vars')
variables_names = [v.name for v in tf.trainable_variables()]
var_values = sess_ftnet.run(variables_names)
#sess_ftnet.close()
print('loaded weight and bias values from fine-tuned model')

######################################
# load unlabeled volumes id numbers to estimate their predictions
unl_list = data_list.train_data(no_of_unl_imgs,comb_unl_imgs)
print('load unlabeled volumes')
print('unl_list',unl_list)
######################################
######################################
def eval_test_imgs_preds(dt, unl_list, sess_ftnet, ae_ft, para_list_fin, test_aug_overlap, test_f1_threshold):
    print('#####################')
    count = 0
    for test_id in unl_list:
        test_id_l = [test_id]
        img_crop_tmp = dt.load_cropped_img_labels(test_id_l, label_present=0)
        print('id, shape', test_id, img_crop_tmp.shape)
        img_crop_re = np.swapaxes(img_crop_tmp, 1, 2)
        img_crop_re = np.swapaxes(img_crop_re, 0, 1)
        # print(img_crop_re.shape)

        if (test_aug_overlap == 0):
            select_mask = 1

        if (test_aug_overlap == 1):
            # create augmentation one (aug1)
            aug_en_list = []
            aug_val_list = []
            img_aug1, aug1_en_list, aug1_val_list = augment_3d_vol(img_crop_re, aug_en_list, aug_val_list, dt, label=0)

            # create augmentation two (aug2)
            # both should not get same flip value!
            #             img_aug2,aug2_en_list,aug2_val_list=augment_3d_vol(img_crop_re,aug_en_list,aug_val_list,dt,label=0)
            while (count >= 0):
                img_aug2, aug2_en_list, aug2_val_list = augment_3d_vol(img_crop_re, aug_en_list, aug_val_list, dt,label=0)
                if (aug1_val_list[0][2] == 1 and aug2_val_list[0][2] == 1):
                    continue
                else:
                    break

            #print('aug_en_lists', aug1_en_list, aug2_en_list)
            #print('aug_val_lists', aug1_val_list, aug2_val_list)
            # apply negative value to applied rotation
            aug1_val_list[0][0] = -aug1_val_list[0][0]
            aug2_val_list[0][0] = -aug2_val_list[0][0]
            # flip its fine -

            # scale: need to do reverse
            if (aug1_en_list[0][1] == 1):
                if (aug1_val_list[0][1] < 1):
                    aug1_val_list[0][1] = 1 + (1 - aug1_val_list[0][1])
                elif (aug1_val_list[0][1] > 1):
                    aug1_val_list[0][1] = 1 + (1 - aug1_val_list[0][1])

            if (aug2_en_list[0][1] == 1):
                if (aug2_val_list[0][1] < 1):
                    aug2_val_list[0][1] = 1 + (1 - aug2_val_list[0][1])
                elif (aug2_val_list[0][1] > 1):
                    aug2_val_list[0][1] = 1 + (1 - aug2_val_list[0][1])

            # print('aug_val_lists',aug1_val_list,aug2_val_list)
            # get prediction for aug1 and aug2
            pred_mask_aug1 = f1_util.calc_pred_mask_batchwise(sess_ftnet, ae_ft, img_aug1)
            pred_mask_aug2 = f1_util.calc_pred_mask_batchwise(sess_ftnet, ae_ft, img_aug2)
            # print('pred_mask_aug shape',pred_mask_aug1.shape,pred_mask_aug2.shape)

            # reverse the augmentations for aug1 and aug2
            pred_aug1_rev, aug1_en_list_rev, aug1_val_list_rev = augment_3d_vol(pred_mask_aug1, aug1_en_list,
                                                                                aug1_val_list, dt, label=1)
            pred_aug2_rev, aug2_en_list_rev, aug2_val_list_rev = augment_3d_vol(pred_mask_aug2, aug2_en_list,
                                                                                aug2_val_list, dt, label=1)

            # f1 score of pred_aug1 and pred_aug2
            pred_aug1 = np.argmax(pred_aug1_rev, axis=-1)
            pred_aug2 = np.argmax(pred_aug2_rev, axis=-1)
            # print('pred_augs',pred_aug1.shape,pred_aug2.shape)

            f1_val = f1_util.calc_f1_score(pred_aug1, pred_aug2)
            f1_mean = np.mean(f1_val[1:cfg.num_classes])
            print('f1 overlap, mean', f1_val, f1_mean)
            # pick this mask if f1_val>0.9 else reject
            if (f1_mean > test_f1_threshold):
                select_mask = 1
            else:
                select_mask = 0

        if (select_mask == 1):
            pred_sf_mask = f1_util.calc_pred_mask_batchwise(sess_ftnet, ae_ft, img_crop_re)
            pred_sf_mask_tmp = np.argmax(pred_sf_mask, axis=-1)
            # print('i',img_crop_re.shape,pred_sf_mask_tmp.shape)

            pred_sf_mask_re = np.swapaxes(pred_sf_mask_tmp, 0, 1)
            pred_sf_mask_re = np.swapaxes(pred_sf_mask_re, 1, 2)
            # print('m',img_crop_tmp.shape,pred_sf_mask_re.shape)

            # can add thresholding / consistency in prediction - Pending implementation
            # Apply CRF on the predicted labels
            unlabeled_labels_tmp_crf = f1_util.calc_crf_op(sess_ftnet, ae_ft, test_img=img_crop_tmp,
                                                           test_mask=pred_sf_mask_re, para_list=para_list_fin)

            if (count == 0):
                unl_img_arr = img_crop_tmp
                unl_lbl_arr = unlabeled_labels_tmp_crf
                count = count + 1
            else:
                unl_img_arr = np.concatenate((unl_img_arr, img_crop_tmp), axis=-1)
                unl_lbl_arr = np.concatenate((unl_lbl_arr, unlabeled_labels_tmp_crf), axis=-1)

        print('#####################')

    print('unlabeled arrays dims', unl_img_arr.shape, unl_lbl_arr.shape)
    return unl_img_arr, unl_lbl_arr


######################################

# In[4]:
unl_img_arr, unl_lbl_arr = eval_test_imgs_preds(dt, unl_list, sess_ftnet, ae_ft, para_list_fin, \
                            parse_config.test_aug_overlap, parse_config.test_f1_threshold)

print('unlabeled arrays dims',unl_img_arr.shape,unl_lbl_arr.shape)


# In[5]:

#close session of fine-tuned seg. network
sess_ftnet.close()


# In[6]:

######################################
# Local loss pre-training
######################################
#define directory to save the pre-training model of decoder with encoder weights frozen (encoder weights obtained from earlier pre-training step)
save_dir=str(cfg.srt_dir)+'/models/'+str(parse_config.dataset)+'/trained_models/pseudo_labels_based_joint_training_on_rand_init/'

#test_aug_overlap
if(parse_config.test_aug_overlap!=0):
    save_dir = str(save_dir) + '/test_aug_overlap_'+str(parse_config.test_aug_overlap)+\
                    '_f1_threshold_'+str(parse_config.test_f1_threshold)+'/'

if(parse_config.data_aug==0):
    save_dir=str(save_dir)+'/no_data_aug/'
else:
    save_dir=str(save_dir)+'/with_data_aug/'

save_dir=str(save_dir)+'local_loss_exp_no_'+str(parse_config.local_loss_exp_no)+'/'

save_dir=str(save_dir)+'temp_fac_'+str(parse_config.temp_fac)+'/'

save_dir=str(save_dir)+'no_of_pos_eles_'+str(parse_config.no_of_pos_eles)+'_no_of_neg_eles_'+str(parse_config.no_of_neg_eles)+'/'

save_dir=str(save_dir)+'/bt_size_'+str(parse_config.bt_size)+'/'

save_dir=str(save_dir)+'/lamda_local_'+str(parse_config.lamda_local)+'/'

save_dir=str(save_dir)+'no_of_unl_imgs_'+str(no_of_unl_imgs)+'_'+str(comb_unl_imgs)+'_no_of_tr_imgs_'+str(parse_config.no_of_tr_imgs)+'_'+str(parse_config.comb_tr_imgs)\
                                        +'_v'+str(parse_config.ver)+'/enc_bbox_dim_'+str(parse_config.bbox_dim)+'_n_iter_'+str(parse_config.n_iter)+'_learning_rate_'+str(parse_config.lr_seg)+'/'

print('save dir ',save_dir)
######################################



# In[7]:

train_list = data_list.train_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
#load saved training data in cropped dimensions directly
print('load train volumes')
train_imgs, train_labels = dt.load_cropped_img_labels(train_list)

#load validation volumes id numbers to save the best model during training
val_list = data_list.val_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
#load val data both in original dimensions and its cropped dimensions
print('loading val volumes')
val_label_orig,val_img_crop,val_label_crop,pixel_val_list=load_val_imgs(val_list,dt,orig_img_dt)

# get test volumes id list
print('get test volumes list')
test_list = data_list.test_data()
######################################

######################################
# Define checkpoint file to save CNN architecture and learnt hyperparameters
checkpoint_filename='train_decoder_wgts_'+str(parse_config.dataset)
logs_path = str(save_dir)+'tensorflow_logs/'
best_model_dir=str(save_dir)+'best_model/'
######################################
pathlib.Path(best_model_dir).mkdir(parents=True, exist_ok=True)


######################################
if parse_config.dataset == 'acdc':
    print('ACDC: set no. of classes used to define contrastive loss per image')
    #minimum no of class labels to be present in each image to be selected in the mini-batch for constrastive loss computation
    no_of_cont_classes=cfg.num_classes
    # minimum no of class labels to be present in each image to be selected in the mini-batch for segmentation loss computation
    no_of_seg_classes=3
    cls_ref = [0, 1, 2, 3]
    print('no_of_cont_classes',no_of_cont_classes,no_of_seg_classes)
elif parse_config.dataset == 'mmwhs':
    print('MMWHS: set no. of classes used to define contrastive loss per image')
    no_of_cont_classes=cfg.num_classes
    no_of_seg_classes=3
    cls_ref = [0,1,2,3,4,5,6,7]
    print('no_of_cont_classes',no_of_cont_classes,no_of_seg_classes)
    print('cls_ref',cls_ref)
elif parse_config.dataset == 'prostate_md':
    print('Prostate_MD: set no. of classes used to define contrastive loss per image')
    no_of_cont_classes=cfg.num_classes
    no_of_seg_classes=2
    cls_ref = [0, 1, 2]
    print('no_of_cont_classes',no_of_cont_classes,no_of_seg_classes)

######################################

# In[8]:

tf.reset_default_graph()
cfg.batch_size_ft=parse_config.bt_size

start = time.time()
print("start time=", time.ctime(start))

ae = model.joint_tr_unet_seg_loss_and_local_cont_loss(learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,\
                                        en_1hot=parse_config.en_1hot,mtask_en=0,temp_fac=parse_config.temp_fac,\
                                        batch_size_ft=cfg.batch_size_ft,local_loss_exp_no=parse_config.local_loss_exp_no,\
                                        no_of_pos_eles=parse_config.no_of_pos_eles, no_of_neg_eles=parse_config.no_of_neg_eles,\
                                        lamda_local=parse_config.lamda_local,no_of_cont_classes=no_of_cont_classes-1)


######################################
end = time.time()
print("end time=", time.ctime(end))
print("diff time=", end - start)


# In[9]:

######################################
#writer for train summary
train_writer = tf.summary.FileWriter(logs_path)
#writer for dice score and val summary
#dsc_writer = tf.summary.FileWriter(logs_path)
val_sum_writer = tf.summary.FileWriter(logs_path)
######################################

ae_rc = model.brit_cont_net(batch_size=cfg.batch_size_ft)
ae_1hot = model.conv_1hot()

# define network/graph to apply random deformations on input images
ae_rd_ld= model.deform_net(batch_size=cfg.mtask_bs)

# define network/graph to apply random contrast and brightness on input images
ae_rc_ld = model.contrast_net(batch_size=cfg.mtask_bs)

#lvl_im=5-parse_config.no_of_decoder_blocks
#size_x,size_y = cfg.img_size_x//np.power(2,lvl_im),cfg.img_size_y//np.power(2,lvl_im)
size_x,size_y = cfg.img_size_x,cfg.img_size_y
print('size_x,y',size_x,size_y)

ae_1hot_mask = model.conv_1hot_mask(size_x,size_y)
ae_1hot_orig = model.conv_1hot()

######################################
# Define session and saver
#sess = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=2)
saver = tf.train.Saver(max_to_keep=2)
######################################


# In[11]:

######################################
#  assign values to all trainable ops of network
assign_op=[]
print('Init of trainable vars')
for new_var in tf.trainable_variables():
    for var, var_val in zip(variables_names, var_values):
        if (str(var) == str(new_var.name) and ('seg_' not in str(new_var.name)) and ('cont_' not in str(new_var.name))):
            #print('match name',new_var.name,var)
            tmp_op=new_var.assign(var_val)
            assign_op.append(tmp_op)

sess.run(assign_op)
print('init done for all the network weights and biases from previous fine-tuned model')
######################################


# In[12]:

######################################
# parameters values set for pre-training of Decoder
#step_val=parse_config.n_iter
start_epoch=0
#n_epochs=step_val
#scale_val = np.power(2,parse_config.no_of_decoder_blocks)/np.power(2,5)
scale_val = 1

max_iter=3
max_iter_count=0
#n_epochs = 5001#2000
tmp_ep = 5001

tr_net_loss_list,tr_seg_loss_list,tr_cont_loss_list=[],[],[]
mean_f1_val_prev=0.0000001
threshold_f1=0.0000001

tr_loss_list,val_loss_list=[],[]
tr_dsc_list,val_dsc_list=[],[]
ep_no_list=[]
loss_least_val=1
f1_mean_least_val=0.0000000001
print_lst=[50,100,200,300,500,1000,2000,3000,4000,6000,8000]

print("time: %s" % time.ctime())


# In[13]:

######################################

for iter_no in range(0, max_iter):

    if(iter_no!=0):
        # restore best model
        saver = tf.train.Saver(max_to_keep=2)
        sess = tf.Session()
        saver.restore(sess, mp_best)

        ######################################
        # Re-estimate pseudo-labels after every 5000 iterations
        unl_img_arr, unl_lbl_arr = eval_test_imgs_preds(dt, unl_list, sess, ae, para_list_fin, \
                                    parse_config.test_aug_overlap,parse_config.test_f1_threshold)
        ######################################

    if (max_iter_count == (max_iter - 1)):
        n_epochs = 301#5001#5000 --pend
        print('last run n_ep', max_iter_count, n_epochs)
    else:
        n_epochs = 301#5001#5000 --pend
        print('early runs n_ep', max_iter_count, n_epochs)

    # Train on this latest model !

    # loop over all the epochs to pre-train the Decoder
    for epoch_i in range(start_epoch,n_epochs):

        epoch_i_overall = (tmp_ep * iter_no) + epoch_i

        # original images batch sampled from unlabeled images
        # labeled set
        ls_img_batch, ls_lbl_batch = shuffle_minibatch([train_imgs, train_labels], batch_size=15*cfg.mtask_bs,labels_present=1)
        # unlabeled set
        ul_img_batch, ul_lbl_batch = shuffle_minibatch([unl_img_arr, unl_lbl_arr], batch_size=30*cfg.batch_size_ft,labels_present=1)
        if (parse_config.local_loss_exp_no == 0):
            #########################
            # Intra pixel match - matching pixel embedding of a class to its same class mean embedding from within the image
            #########################
            ls_new_labels=[]
            ls_img_batch_re=[]

            #pos_cls_list_np = np.zeros((2 * cfg.batch_size_ft, no_of_cont_classes - 1), dtype=np.int32)
            count, count_cls_l = 0, 0
            for index in range(ls_lbl_batch.shape[0]):
                lbl = np.squeeze(ls_lbl_batch[index,...])
                lbl_re= transform.rescale(lbl, scale_val, order=0, preserve_range=True, mode='constant')
                #print(np.unique(lbl_re),np.unique(lbl))
                lbl_uniq = np.unique(lbl_re, return_counts=1)

                equal_arrays = np.array_equal(np.asarray(lbl_uniq[0]), np.asarray(cls_ref))
                if (len(lbl_uniq[0]) == no_of_cont_classes and equal_arrays==True):
                    ls_new_labels.append(lbl_re[...])
                    ls_img_batch_re.append(ls_img_batch[index])
                    #pos_cls_list_np[count_cls_l] = lbl_uniq[0][1:no_of_cont_classes]
                    count = count + 1
                    #count_cls_l = count_cls_l + 1
                if(count==(cfg.batch_size_ft//2)):
                    break

            if (count != (cfg.batch_size_ft//2)):
                continue

            ul_new_labels = []
            ul_img_batch_re = []
            count=0
            for index in range(ul_lbl_batch.shape[0]):
                lbl = np.squeeze(ul_lbl_batch[index, ...])
                lbl_re = transform.rescale(lbl, scale_val, order=0, preserve_range=True, mode='constant')
                # print(np.unique(lbl_re),np.unique(lbl))
                lbl_uniq = np.unique(lbl_re, return_counts=1)

                equal_arrays = np.array_equal(np.asarray(lbl_uniq[0]), np.asarray(cls_ref))
                if (len(lbl_uniq[0]) == no_of_cont_classes and equal_arrays == True):
                    ul_new_labels.append(lbl_re[...])
                    ul_img_batch_re.append(ul_img_batch[index])
                    # pos_cls_list_np[count_cls_l] = lbl_uniq[0][1:no_of_cont_classes]
                    count = count + 1
                if (count == (cfg.batch_size_ft//2)):
                    break

            if (count != (cfg.batch_size_ft//2)):
                continue

            img_batch_re = np.concatenate((np.asarray(ls_img_batch_re), np.asarray(ul_img_batch_re)), axis=0)
            lbl_rescaled = np.concatenate((np.asarray(ls_new_labels), np.asarray(ul_new_labels)), axis=0)

            # make 2 different sets of image, label pairs from this chosen batch.
            # random intensity aug - random contrast + brightness
            # Set 1 - random intensity aug on Labeled set
            color_batch1 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: img_batch_re})
            # Set 2 - random intensity aug on UNLabeled set
            color_batch2 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: img_batch_re})

            # Stitch these 2 augmented sets into 1 batch for pre-training
            img_cat_batch = np.concatenate([color_batch1[0:cfg.batch_size_ft], color_batch2[0:cfg.batch_size_ft]], axis=0)
            lbl_cat_batch = np.concatenate([lbl_rescaled[0:cfg.batch_size_ft], lbl_rescaled[0:cfg.batch_size_ft]], axis=0)

            #get 1-hot encoding of masks
            lbl_cat_batch = sess.run(ae_1hot_mask['y_tmp_1hot'],feed_dict={ae_1hot_mask['y_tmp']:lbl_cat_batch})

            #calculate pos and neg indices of masks
            net_pos_ele1_arr,net_pos_ele2_arr,net_neg_ele1_arr,net_neg_ele2_arr=\
                calc_pos_neg_index(lbl_cat_batch,cfg.batch_size_ft,cfg.num_classes,parse_config.no_of_pos_eles,parse_config.no_of_neg_eles)

            pos_arr= np.concatenate((net_pos_ele1_arr,net_pos_ele2_arr),axis=0)
            neg_arr= np.concatenate((net_neg_ele1_arr,net_neg_ele2_arr),axis=0)
            #print(pos_arr.shape, neg_arr.shape)

        elif (parse_config.local_loss_exp_no == 1):
            #########################
            # Inter pixel match - matching pixel embedding of a class to the same class mean embedding from a different image and from same image
            #########################
            ls_new_labels = []
            ls_img_batch_re = []
            count = 0
            for index in range(ls_lbl_batch.shape[0]):
                lbl = np.squeeze(ls_lbl_batch[index, ...])
                lbl_re = transform.rescale(lbl, scale_val, order=0, preserve_range=True, mode='constant')
                # print(np.unique(lbl_re),np.unique(lbl))
                lbl_uniq = np.unique(lbl_re, return_counts=1)

                equal_arrays = np.array_equal(np.asarray(lbl_uniq[0]), np.asarray(cls_ref))
                if (len(lbl_uniq[0]) == no_of_cont_classes and equal_arrays == True):
                    ls_new_labels.append(lbl_re[...])
                    ls_img_batch_re.append(ls_img_batch[index])
                    count = count + 1
                if (count == (cfg.batch_size_ft // 2)):
                    break

            if (count != (cfg.batch_size_ft // 2)):
                continue

            ul_new_labels = []
            ul_img_batch_re = []
            count = 0
            for index in range(ul_lbl_batch.shape[0]):
                lbl = np.squeeze(ul_lbl_batch[index, ...])
                lbl_re = transform.rescale(lbl, scale_val, order=0, preserve_range=True, mode='constant')
                # print(np.unique(lbl_re),np.unique(lbl))
                lbl_uniq = np.unique(lbl_re, return_counts=1)

                equal_arrays = np.array_equal(np.asarray(lbl_uniq[0]), np.asarray(cls_ref))
                if (len(lbl_uniq[0]) == no_of_cont_classes and equal_arrays == True):
                    ul_new_labels.append(lbl_re[...])
                    ul_img_batch_re.append(ul_img_batch[index])
                    count = count + 1
                if (count == (cfg.batch_size_ft // 2)):
                    break

            if (count != (cfg.batch_size_ft // 2)):
                continue

            ls_batch_re, ul_batch_re = np.asarray(ls_img_batch_re), np.asarray(ul_img_batch_re)
            ls_lbl_rescaled, ul_lbl_rescaled = np.asarray(ls_new_labels), np.asarray(ul_new_labels)

            net_img_batch1_re = np.zeros((cfg.batch_size_ft, cfg.img_size_x, cfg.img_size_y, 1))
            net_img_batch2_re = np.zeros((cfg.batch_size_ft, cfg.img_size_x, cfg.img_size_y, 1))
            net_lbl_batch1_re = np.zeros((cfg.batch_size_ft, size_x, size_y))
            net_lbl_batch2_re = np.zeros((cfg.batch_size_ft, size_x, size_y))

            idx_count = 0

            idx_count, index = 0, 0
            net_img_batch1_re[idx_count] = ls_batch_re[index]
            net_img_batch1_re[idx_count + 1] = ls_batch_re[index + 1]
            net_lbl_batch1_re[idx_count] = ls_lbl_rescaled[index]
            net_lbl_batch1_re[idx_count + 1] = ls_lbl_rescaled[index + 1]
            net_img_batch2_re[idx_count] = ls_batch_re[index]
            net_img_batch2_re[idx_count + 1] = ul_batch_re[index + 1]
            net_lbl_batch2_re[idx_count] = ls_lbl_rescaled[index]
            net_lbl_batch2_re[idx_count + 1] = ul_lbl_rescaled[index + 1]

            idx_count, index = 2, 2
            net_img_batch1_re[idx_count] = ls_batch_re[index]
            net_img_batch1_re[idx_count + 1] = ls_batch_re[index + 1]
            net_lbl_batch1_re[idx_count] = ls_lbl_rescaled[index]
            net_lbl_batch1_re[idx_count + 1] = ls_lbl_rescaled[index + 1]
            net_img_batch2_re[idx_count] = ls_batch_re[index]
            net_img_batch2_re[idx_count + 1] = ul_batch_re[index + 1]
            net_lbl_batch2_re[idx_count] = ls_lbl_rescaled[index]
            net_lbl_batch2_re[idx_count + 1] = ul_lbl_rescaled[index + 1]

            idx_count, index = 4, 4
            net_img_batch1_re[idx_count] = ls_batch_re[index]
            net_lbl_batch1_re[idx_count] = ls_lbl_rescaled[index]
            net_img_batch2_re[idx_count] = ul_batch_re[index]
            net_lbl_batch2_re[idx_count] = ul_lbl_rescaled[index]

            idx_count, index = 5, 0
            net_img_batch1_re[idx_count] = ul_batch_re[index]
            net_img_batch1_re[idx_count + 1] = ul_batch_re[index]
            net_lbl_batch1_re[idx_count] = ul_lbl_rescaled[index]
            net_lbl_batch1_re[idx_count + 1] = ul_lbl_rescaled[index]
            net_img_batch2_re[idx_count] = ul_batch_re[index]
            net_img_batch2_re[idx_count + 1] = ul_batch_re[index + 1]
            net_lbl_batch2_re[idx_count] = ul_lbl_rescaled[index]
            net_lbl_batch2_re[idx_count + 1] = ul_lbl_rescaled[index + 1]

            idx_count, index = 7, 2
            net_img_batch1_re[idx_count] = ul_batch_re[index]
            net_img_batch1_re[idx_count + 1] = ul_batch_re[index]
            net_lbl_batch1_re[idx_count] = ul_lbl_rescaled[index]
            net_lbl_batch1_re[idx_count + 1] = ul_lbl_rescaled[index]
            net_img_batch2_re[idx_count] = ul_batch_re[index]
            net_img_batch2_re[idx_count + 1] = ul_batch_re[index + 1]
            net_lbl_batch2_re[idx_count] = ul_lbl_rescaled[index]
            net_lbl_batch2_re[idx_count + 1] = ul_lbl_rescaled[index + 1]

            idx_count, index = 9, 3
            net_img_batch1_re[idx_count] = ul_batch_re[index]
            net_lbl_batch1_re[idx_count] = ul_lbl_rescaled[index]
            net_img_batch2_re[idx_count] = ul_batch_re[index+1]
            net_lbl_batch2_re[idx_count] = ul_lbl_rescaled[index+1]

            # Make a batch of labeled and unlabeled images and respective masks
            # random intensity aug - random contrast + brightness
            # Set 1 - random intensity aug on Labeled set
            color_batch1 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: net_img_batch1_re})
            # Set 2 - random intensity aug on UNLabeled set
            color_batch2 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: net_img_batch2_re})

            # Stitch these 2 augmented sets into 1 batch for pre-training
            img_cat_batch = np.concatenate([color_batch1[0:cfg.batch_size_ft], color_batch2[0:cfg.batch_size_ft]], axis=0)
            lbl_cat_batch = np.concatenate([net_lbl_batch1_re[0:cfg.batch_size_ft], net_lbl_batch2_re[0:cfg.batch_size_ft]],axis=0)

            # get 1-hot encoding of masks
            lbl_cat_batch = sess.run(ae_1hot_mask['y_tmp_1hot'], feed_dict={ae_1hot_mask['y_tmp']: lbl_cat_batch})

            # calculate pos and neg indices of masks
            net_pos_ele1_arr, net_pos_ele2_arr, net_neg_ele1_arr, net_neg_ele2_arr = \
                calc_pos_neg_index(lbl_cat_batch, cfg.batch_size_ft, cfg.num_classes, parse_config.no_of_pos_eles,
                                   parse_config.no_of_neg_eles)

            pos_arr = np.concatenate((net_pos_ele1_arr, net_pos_ele2_arr), axis=0)
            neg_arr = np.concatenate((net_neg_ele1_arr, net_neg_ele2_arr), axis=0)
            # print(pos_arr.shape, neg_arr.shape)

        # calculate pos and neg indices of masks
        pos_count_arr1 = np.count_nonzero(np.where(net_pos_ele1_arr != 1000))
        pos_count_arr2 = np.count_nonzero(np.where(net_pos_ele2_arr != 1000))
        # print(pos_count)

        neg_count_arr1 = np.count_nonzero(np.where(net_neg_ele1_arr != 1000))
        neg_count_arr2 = np.count_nonzero(np.where(net_neg_ele2_arr != 1000))

        if (pos_count_arr1 == 0 or pos_count_arr2 == 0 or neg_count_arr1 == 0 or neg_count_arr2 == 0):
            print('empty_arr count', pos_count_arr1, pos_count_arr2, neg_count_arr1, neg_count_arr2)
            print('epoch_i', epoch_i_overall)
            continue

        ##################################
        # created labeled examples batch
        ##################################
        # Apply affine transformations
        ls_new_labels = []
        ls_img_batch_re = []
        count = 0
        for index in range(ls_lbl_batch.shape[0]):
            lbl_re = np.squeeze(ls_lbl_batch[index, ...])
            #lbl_re = transform.rescale(lbl, scale_val, order=0, preserve_range=True, mode='constant')
            # print(np.unique(lbl_re),np.unique(lbl))
            lbl_uniq = np.unique(lbl_re, return_counts=1)
            # if (len(lbl_uniq[0]) >= cfg.num_classes):
            if (len(lbl_uniq[0]) >= no_of_seg_classes):
                ls_new_labels.append(lbl_re[...])
                ls_img_batch_re.append(ls_img_batch[index])
                count = count + 1
            if (count == (cfg.mtask_bs)):
                break

        if (count != (cfg.mtask_bs)):
            continue

        ls_img_batch = np.asarray(ls_img_batch_re)
        ls_lbl_batch = np.asarray(ls_new_labels)

        ld_img_batch, ld_label_batch = augmentation_function([ls_img_batch[0:cfg.mtask_bs], ls_lbl_batch[0:cfg.mtask_bs]], dt)
        if (parse_config.rd_en == 1 or parse_config.ri_en == 1):
            # Apply random augmentations - random deformations + random contrast & brightness values
            ld_img_batch, ld_label_batch = create_rand_augs(cfg, parse_config, sess, ae_rd_ld, ae_rc_ld, ld_img_batch,ld_label_batch)

        ##################################
        # concatenate labeled examples batch and unlabeled examples batch
        img_cat_batch = np.concatenate([ld_img_batch, img_cat_batch], axis=0)

        ##################################

        #Run optimizer to update network weights for the joint training
        try:
            cont_loss, num_i1_ss, num_i2_ss, den_i1_ss, den_i2_ss = \
                        sess.run([ae['cont_loss_cost'], ae['num_i1_ss'], ae['num_i2_ss'], ae['den_i1_ss'], ae['den_i2_ss']], \
                         feed_dict={ae['x']:img_cat_batch,ae['y_l']:ld_label_batch,\
                                    ae['y_l_reg']:lbl_cat_batch,ae['pos_indx']:pos_arr,\
                            ae['neg_indx']:neg_arr,ae['train_phase']:True})
            #reg_loss=reg_loss[0]
            #if(math.isnan(reg_loss)==False):
            if (math.isnan(cont_loss) == True or cont_loss == 0):
                print('continue,epoch_i', epoch_i_overall)
                continue

            if (num_i1_ss == 0 or num_i2_ss == 0 or den_i1_ss == 0 or den_i2_ss == 0):
                print('continue,epoch_i', epoch_i_overall)
                continue

            if (math.isnan(cont_loss) == False and cont_loss != 0 and math.isnan(num_i1_ss) == False\
                and math.isnan(num_i2_ss) == False  and math.isnan(den_i1_ss) == False  and math.isnan(den_i2_ss) == False):
                train_summary,tr_loss,seg_loss,cont_loss,_=sess.run([ae['train_summary'],ae['net_cost'],ae['seg_cost'],\
                                                ae['cont_loss_cost'],ae['optimizer_unet_all']],\
                                         feed_dict={ae['x']:img_cat_batch,ae['y_l']:ld_label_batch,\
                                                    ae['y_l_reg']:lbl_cat_batch,ae['pos_indx']:pos_arr,\
                                ae['neg_indx']:neg_arr,ae['train_phase']:True})
        except:
            print('update both enc and dec: exception at Ep-',epoch_i_overall)
            continue

        if(epoch_i%cfg.val_step_update==0 and math.isnan(cont_loss)==False):
            train_writer.add_summary(train_summary, epoch_i_overall)
            train_writer.flush()

            tr_net_loss_list.append(np.mean(tr_loss))
            tr_seg_loss_list.append(np.mean(seg_loss))
            tr_cont_loss_list.append(np.mean(cont_loss))

            print('epoch_i, net, seg, cont losses - ', epoch_i_overall, np.mean(tr_loss),np.mean(seg_loss),np.mean(cont_loss))

        if(epoch_i%cfg.val_step_update==0 and math.isnan(cont_loss)==False):
            # Measure validation volumes accuracy in Dice score (DSC) and evaluate validation loss
            # Save the model with the best DSC over validation volumes.
            mean_f1_val_prev,mp_best,mean_total_cost_val,mean_f1=f1_util.track_val_dsc(sess,ae,ae_1hot_orig,saver,mean_f1_val_prev,threshold_f1,\
                                        best_model_dir,val_list,val_img_crop,val_label_crop,val_label_orig,pixel_val_list,\
                                        checkpoint_filename,epoch_i_overall,en_1hot_val=parse_config.en_1hot)

            tr_y_pred=sess.run(ae['y_pred'],feed_dict={ae['x']:ld_img_batch,ae['y_l']:ld_label_batch,ae['train_phase']:False})

            if(parse_config.en_1hot==1):
                tr_accu=f1_util.calc_f1_score(np.argmax(tr_y_pred,axis=-1),np.argmax(ld_label_batch,-1))
            else:
                tr_accu=f1_util.calc_f1_score(np.argmax(tr_y_pred,axis=-1),ld_label_batch)

            tr_dsc_list.append(np.mean(tr_accu))
            val_dsc_list.append(mean_f1)
            tr_loss_list.append(np.mean(seg_loss))
            val_loss_list.append(np.mean(mean_total_cost_val))

            print('epoch_i,loss,f1_val',epoch_i_overall,np.mean(seg_loss),mean_f1_val_prev,mean_f1)

            #Compute and save validation image dice & loss summary
            val_summary_msg = sess.run(ae['val_summary'], feed_dict={ae['mean_dice']: mean_f1, ae['val_totalc']:mean_total_cost_val})
            val_sum_writer.add_summary(val_summary_msg, epoch_i_overall)
            val_sum_writer.flush()

            if(np.mean(mean_f1_val_prev)>f1_mean_least_val):
                f1_mean_least_val=mean_f1_val_prev
                ep_no_list.append(epoch_i_overall)


        if ((epoch_i==n_epochs-1)):
            # model saved at the last epoch of training
            mp = str(save_dir) + str(checkpoint_filename) + '_epochs_' + str(epoch_i_overall) + ".ckpt"
            saver.save(sess, mp)

            try:
                mp_best
            except NameError:
                mp_best = mp
            try:
                mean_f1
            except NameError:
                mean_f1=0
            mean_f1_val_prev = mean_f1

        # if (epoch_i_overall in print_lst):
        #     f1_util.plt_seg_loss([tr_net_loss_list], save_dir, title_str='joint_train_net_iter_no_'+str(iter_no),plt_name='tr_net_loss_iter_no_'+str(iter_no), ep_no=epoch_i_overall)
        #     f1_util.plt_seg_loss([tr_seg_loss_list], save_dir, title_str='joint_train_net_iter_no_'+str(iter_no),plt_name='tr_seg_loss_iter_no_'+str(iter_no), ep_no=epoch_i_overall)
        #     f1_util.plt_seg_loss([tr_cont_loss_list], save_dir, title_str='joint_train_net_iter_no_'+str(iter_no),plt_name='tr_local_cont_loss_iter_no_'+str(iter_no), ep_no=epoch_i_overall)
    if (epoch_i_overall in print_lst):
        f1_util.plt_seg_loss([tr_net_loss_list], save_dir, title_str='joint_train_net_iter_no_'+str(iter_no),plt_name='tr_net_loss_iter_no_'+str(iter_no), ep_no=epoch_i_overall)
        f1_util.plt_seg_loss([tr_seg_loss_list], save_dir, title_str='joint_train_net_iter_no_'+str(iter_no),plt_name='tr_seg_loss_iter_no_'+str(iter_no), ep_no=epoch_i_overall)
        f1_util.plt_seg_loss([tr_cont_loss_list], save_dir, title_str='joint_train_net_iter_no_'+str(iter_no),plt_name='tr_local_cont_loss_iter_no_'+str(iter_no), ep_no=epoch_i_overall)

    print('mp_best', epoch_i_overall, mp_best)

    max_iter_count = max_iter_count + 1
    try:
        mp_best = get_max_chkpt_file(save_dir)
        print('iter mp_best', mp_best)
    except:
        mp_best = get_chkpt_file(save_dir)
        print('iter mp_last_ep', mp_best)

    sess.close()


######################################
# Plot the training loss of train set over all the iterations of training.
f1_util.plt_seg_loss([tr_net_loss_list],save_dir,title_str='joint_train_net_iter_no_',plt_name='tr_net_loss',ep_no=epoch_i_overall)
f1_util.plt_seg_loss([tr_seg_loss_list],save_dir,title_str='joint_train_net_iter_no_',plt_name='tr_seg_loss',ep_no=epoch_i_overall)
f1_util.plt_seg_loss([tr_cont_loss_list],save_dir,title_str='joint_train_net_iter_no_',plt_name='tr_local_cont_loss',ep_no=epoch_i_overall)
######################################
# Plot the training, validation (val) loss and DSC score of training & val sets over all the iterations of training.
f1_util.plt_seg_loss([tr_loss_list,val_loss_list],save_dir,title_str='joint_training',plt_name='train_seg_loss',ep_no=epoch_i_overall)
f1_util.plt_seg_loss([tr_dsc_list,val_dsc_list],save_dir,title_str='joint_training',plt_name='train_dsc_score',ep_no=epoch_i_overall)
######################################

print("time: %s" % time.ctime())
print('Done')

sess.close()
######################################

######################################
# find best model checkpoint over all epochs and restore it
try:
    mp_best = get_max_chkpt_file(save_dir)
    print('mp_best', mp_best)
except:
    mp_best = get_chkpt_file(save_dir)
    print('mp_last_ep', mp_best)

saver = tf.train.Saver()
#sess = tf.Session(config=config)
sess = tf.Session()
saver.restore(sess, mp_best)
print("Model restored")
######################################
# infer predictions over test volumes from the best model saved during training
struct_name=cfg.struct_name
save_dir_tmp=save_dir+'/test_set_predictions/'
f1_util.test_set_predictions(test_list,sess,ae,dt,orig_img_dt,save_dir_tmp,struct_name)

sess.close()
tf.reset_default_graph()
######################################
