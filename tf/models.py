import tensorflow as tf
import numpy as np
import time

# Load layers and losses
from layers_bn import layersObj
layers = layersObj()

from losses import lossObj
loss = lossObj()

class modelObj:
    def __init__(self,cfg,override_num_classes=0):
        self.img_size_x=cfg.img_size_x
        self.img_size_y=cfg.img_size_y
        self.num_classes=cfg.num_classes
        self.num_channels=cfg.num_channels

        self.interp_val = cfg.interp_val
        self.img_size_flat=cfg.img_size_flat
        self.batch_size=cfg.batch_size_ft
        self.dataset_name=cfg.dataset_name

        self.mtask_bs=cfg.mtask_bs

        if(override_num_classes==1):
            self.num_classes=2

    def conv_1hot_mask(self,size_x,size_y):
        # To compute the 1-hot encoding of input mask to number of classes
        # placeholders for the network
        y_tmp = tf.placeholder(tf.int32, shape=[None, size_x,size_y], name='y_tmp')

        y_tmp_1hot = tf.one_hot(y_tmp,depth=self.num_classes)
        return {'y_tmp':y_tmp,'y_tmp_1hot':y_tmp_1hot}

    def conv_1hot(self):
        # To compute the 1-hot encoding of input mask to number of classes
        # placeholders for the network
        y_tmp = tf.placeholder(tf.int32, shape=[None, self.img_size_x, self.img_size_y], name='y_tmp')

        y_tmp_1hot = tf.one_hot(y_tmp,depth=self.num_classes)
        return {'y_tmp':y_tmp,'y_tmp_1hot':y_tmp_1hot}

    def deform_net(self,batch_size):
        # To apply random deformations on the input image and segmentation mask

        # placeholders for the network
        x_tmp = tf.placeholder(tf.float32, shape=[batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')
        v_tmp = tf.placeholder(tf.float32, shape=[batch_size, self.img_size_x, self.img_size_y, 2], name='v_tmp')
        y_tmp = tf.placeholder(tf.int32, shape=[batch_size, self.img_size_x, self.img_size_y], name='y_tmp')

        y_tmp_1hot = tf.one_hot(y_tmp,depth=self.num_classes)
        w_tmp = tf.contrib.image.dense_image_warp(image=x_tmp,flow=v_tmp,name='dense_image_warp_tmp')
        w_tmp_1hot = tf.contrib.image.dense_image_warp(image=y_tmp_1hot,flow=v_tmp,name='dense_image_warp_tmp_1hot')

        return {'x_tmp':x_tmp,'flow_v':v_tmp,'deform_x':w_tmp,'y_tmp':y_tmp,'y_tmp_1hot':y_tmp_1hot,'deform_y_1hot':w_tmp_1hot}

    def contrast_net(self,batch_size):
        # To apply random contrast and brightness (random intensity transformations) on the input image (Fine-training stage)

        # placeholders for the network
        x_tmp = tf.placeholder(tf.float32, shape=[batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')

        rd_cont = tf.image.random_contrast(x_tmp,lower=0.8,upper=1.2,seed=1)
        rd_brit = tf.image.random_brightness(x_tmp,max_delta=0.1,seed=1)
        c_ind=np.arange(0,int(batch_size/2),dtype=np.int32)
        b_ind=np.arange(int(batch_size/2),int(batch_size),dtype=np.int32)

        rd_fin = tf.concat((tf.gather(rd_cont,c_ind),tf.gather(rd_brit,b_ind)),axis=0)
        return {'x_tmp':x_tmp,'rd_fin':rd_fin,'rd_cont':rd_cont,'rd_brit':rd_brit}

    def brit_cont_net(self,batch_size):
        # To apply random contrast and brightness (random intensity transformations) on the input image (Pre-training stages)

        # placeholders for the network
        x_tmp = tf.placeholder(tf.float32, shape=[batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')

        # brightness + contrast changes final image
        rd_brit = tf.image.random_brightness(x_tmp,max_delta=0.3,seed=1)
        rd_cont = tf.image.random_contrast(rd_brit,lower=0.7,upper=1.3,seed=1)
        rd_fin=tf.clip_by_value(rd_cont,0,1.5)

        return {'x_tmp':x_tmp,'rd_fin':rd_fin,'rd_cont':rd_cont,'rd_brit':rd_brit}


    def cos_sim(self,vec_a,vec_b,temp_fac):
        # To compute the cosine similarity score of the input 2 vectors scaled by temparature factor

        norm_vec_a = tf.nn.l2_normalize(vec_a,axis=-1)
        norm_vec_b = tf.nn.l2_normalize(vec_b,axis=-1)
        #cos_sim_val=tf.multiply(norm_vec_a,norm_vec_b)/scale_fac
        cos_sim_val=tf.linalg.matmul(norm_vec_a,norm_vec_b,transpose_b=True)/temp_fac
        return cos_sim_val

    def encoder_network(self,x,train_phase,no_filters,encoder_list_return=0):
        # Define the Encoder Network

        #layers list for skip connections
        enc_layers_list=[]
        ############################################
        # U-Net like Network
        ############################################
        # Encoder - Downsampling Path
        ############################################
        # two 3x3 conv and 1 maxpool
        # Level 1
        enc_c1_a = layers.conv2d_layer(ip_layer=x, name='enc_c1_a', num_filters=no_filters[1], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c1_b = layers.conv2d_layer(ip_layer=enc_c1_a, name='enc_c1_b', num_filters=no_filters[1], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c1_pool = layers.max_pool_layer2d(enc_c1_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c1_pool')
        enc_layers_list.append(enc_c1_b)

        # Level 2
        enc_c2_a = layers.conv2d_layer(ip_layer=enc_c1_pool, name='enc_c2_a', num_filters=no_filters[2], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c2_b = layers.conv2d_layer(ip_layer=enc_c2_a, name='enc_c2_b', num_filters=no_filters[2], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c2_pool = layers.max_pool_layer2d(enc_c2_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c2_pool')
        enc_layers_list.append(enc_c2_b)

        # Level 3
        enc_c3_a = layers.conv2d_layer(ip_layer=enc_c2_pool, name='enc_c3_a', num_filters=no_filters[3], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c3_b = layers.conv2d_layer(ip_layer=enc_c3_a, name='enc_c3_b', num_filters=no_filters[3], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c3_pool = layers.max_pool_layer2d(enc_c3_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c3_pool')
        enc_layers_list.append(enc_c3_b)

        # Level 4
        enc_c4_a = layers.conv2d_layer(ip_layer=enc_c3_pool, name='enc_c4_a', num_filters=no_filters[4], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c4_b = layers.conv2d_layer(ip_layer=enc_c4_a, name='enc_c4_b', num_filters=no_filters[4], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c4_pool = layers.max_pool_layer2d(enc_c4_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c4_pool')
        enc_layers_list.append(enc_c4_b)

        # Level 5 - 2x Conv
        enc_c5_a = layers.conv2d_layer(ip_layer=enc_c4_pool, name='enc_c5_a', num_filters=no_filters[5], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c5_b = layers.conv2d_layer(ip_layer=enc_c5_a, name='enc_c5_b', num_filters=no_filters[5], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c5_pool = layers.max_pool_layer2d(enc_c5_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c5_pool')
        enc_layers_list.append(enc_c5_b)

        # Level 6 - 2x Conv
        enc_c6_a = layers.conv2d_layer(ip_layer=enc_c5_pool, name='enc_c6_a', num_filters=no_filters[5], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c6_b = layers.conv2d_layer(ip_layer=enc_c6_a, name='enc_c6_b', num_filters=no_filters[5], use_relu=True,use_batch_norm=True, training_phase=train_phase)

        if(encoder_list_return==1):
            return enc_c6_b,enc_layers_list
        else:
            return enc_c6_b

    def decoder_network(self, enc_c6_b, train_phase, no_filters, enc_layers_list, fs_de):
        ###################################
        # skip-connection layers from encoder
        enc_c1_b, enc_c2_b, enc_c3_b, enc_c4_b, enc_c5_b = enc_layers_list[0], enc_layers_list[1], enc_layers_list[2], enc_layers_list[3], enc_layers_list[4]
        ###################################
        # Decoder network - Upsampling Path
        ###################################
        # one upsampling layer with a factor of 2, a skip connection from encoder to this upsampling layer output, lastly followed by two 3x3 convs
        scale_fac = 2
        dec_c6_up = layers.upsample_layer(ip_layer=enc_c6_b, method=self.interp_val, scale_factor=int(scale_fac))
        # print('dec 2 large up',dec_c6_up)
        dec_dc6 = layers.conv2d_layer(ip_layer=dec_c6_up, name='dec_dc6', kernel_size=(fs_de, fs_de),num_filters=no_filters[5], use_relu=True, use_batch_norm=True,training_phase=train_phase)
        dec_cat_c6 = tf.concat((dec_dc6, enc_c5_b), axis=3, name='dec_cat_c6')
        dec_c5_a = layers.conv2d_layer(ip_layer=dec_cat_c6, name='dec_c5_a', num_filters=no_filters[5], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        dec_c5_b = layers.conv2d_layer(ip_layer=dec_c5_a, name='dec_c5_b', num_filters=no_filters[5], use_relu=True,use_batch_norm=True, training_phase=train_phase)

        dec_c5_up = layers.upsample_layer(ip_layer=dec_c5_b, method=self.interp_val, scale_factor=int(scale_fac))
        # print('dec large up',dec_c6_up,dec_c5_up)
        dec_dc5 = layers.conv2d_layer(ip_layer=dec_c5_up, name='dec_dc5', kernel_size=(fs_de, fs_de),num_filters=no_filters[4], use_relu=True, use_batch_norm=True,training_phase=train_phase)
        dec_cat_c5 = tf.concat((dec_dc5, enc_c4_b), axis=3, name='dec_cat_c5')
        dec_c4_a = layers.conv2d_layer(ip_layer=dec_cat_c5, name='dec_c4_a', num_filters=no_filters[4], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        dec_c4_b = layers.conv2d_layer(ip_layer=dec_c4_a, name='dec_c4_b', num_filters=no_filters[4], use_relu=True,use_batch_norm=True, training_phase=train_phase)

        # Level 4
        dec_up4 = layers.upsample_layer(ip_layer=dec_c4_b, method=self.interp_val, scale_factor=scale_fac)
        dec_dc4 = layers.conv2d_layer(ip_layer=dec_up4, name='dec_dc4', kernel_size=(fs_de, fs_de),num_filters=no_filters[3], use_relu=True, use_batch_norm=True,training_phase=train_phase)
        dec_cat_c4 = tf.concat((dec_dc4, enc_c3_b), axis=3, name='dec_cat_c4')
        dec_c3_a = layers.conv2d_layer(ip_layer=dec_cat_c4, name='dec_c3_a', num_filters=no_filters[3], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        dec_c3_b = layers.conv2d_layer(ip_layer=dec_c3_a, name='dec_c3_b', num_filters=no_filters[3], use_relu=True,use_batch_norm=True, training_phase=train_phase)

        # Level 3
        dec_up3 = layers.upsample_layer(ip_layer=dec_c3_b, method=self.interp_val, scale_factor=scale_fac)
        dec_dc3 = layers.conv2d_layer(ip_layer=dec_up3, name='dec_dc3', kernel_size=(fs_de, fs_de),num_filters=no_filters[2], use_relu=True, use_batch_norm=True,training_phase=train_phase)
        dec_cat_c3 = tf.concat((dec_dc3, enc_c2_b), axis=3, name='dec_cat_c3')
        dec_c2_a = layers.conv2d_layer(ip_layer=dec_cat_c3, name='dec_c2_a', num_filters=no_filters[2], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        dec_c2_b = layers.conv2d_layer(ip_layer=dec_c2_a, name='dec_c2_b', num_filters=no_filters[2], use_relu=True,use_batch_norm=True, training_phase=train_phase)

        # Level 2
        dec_up2 = layers.upsample_layer(ip_layer=dec_c2_b, method=self.interp_val, scale_factor=scale_fac)
        dec_dc2 = layers.conv2d_layer(ip_layer=dec_up2, name='dec_dc2', kernel_size=(fs_de, fs_de),num_filters=no_filters[1], use_relu=True, use_batch_norm=True,training_phase=train_phase)
        dec_cat_c2 = tf.concat((dec_dc2, enc_c1_b), axis=3, name='dec_cat_c2')
        dec_c1_a = layers.conv2d_layer(ip_layer=dec_cat_c2, name='dec_c1_a', num_filters=no_filters[1], use_relu=True,use_batch_norm=True, training_phase=train_phase)

        return dec_c1_a

    def seg_unet(self,learn_rate_seg=0.001,dsc_loss=2,en_1hot=0,mtask_en=1,fs_de=2):
        # Define the U-Net (Encoder & Decoder Network) to segment the input image

        # No of channels in each level of encoder / decoder
        no_filters=[1, 16, 32, 64, 128, 128]

        if(self.num_classes==2):
            class_weights = tf.constant([[0.1, 0.9]],name='class_weights')
        elif(self.num_classes==3):
            class_weights = tf.constant([[0.1, 0.45, 0.45]],name='class_weights')
        elif(self.num_classes==4):
            class_weights = tf.constant([[0.1, 0.3, 0.3, 0.3]],name='class_weights')
        elif (self.num_classes==8):
            class_weights = tf.constant([[0.09, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13]], name='class_weights')

        num_channels=self.num_channels

        # placeholders for the network
        # Inputs
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y, num_channels], name='x')
        if(en_1hot==1):
            y_l = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y,self.num_classes], name='y_l')
        else:
            y_l = tf.placeholder(tf.int32, shape=[None, self.img_size_x, self.img_size_y], name='y_l')
        train_phase = tf.placeholder(tf.bool, name='train_phase')

        if(en_1hot==0):
            y_l_onehot=tf.one_hot(y_l,depth=self.num_classes)
        else:
            y_l_onehot=y_l
        #print('x,y_l_onehot',x,y_l_onehot)

        ###################################
        # Common Encoder network
        ###################################
        # Last layer from Encoder network
        enc_c6_b,enc_layers_list = self.encoder_network(x, train_phase, no_filters,encoder_list_return=1)

        ###################################
        # skip-connection layers from encoder
        #enc_c1_b,enc_c2_b,enc_c3_b,enc_c4_b,enc_c5_b = enc_layers_list[0],enc_layers_list[1],enc_layers_list[2],enc_layers_list[3],enc_layers_list[4]

        ###################################
        # Common Decoder Network
        ###################################
        dec_c1_a = self.decoder_network(enc_c6_b, train_phase, no_filters, enc_layers_list, fs_de)

        ###################################
        # Segmentation specific layers
        ###################################
        # g_\xi - small network with few convolutions for segmentation loss computation
        seg_c1_a = layers.conv2d_layer(ip_layer=dec_c1_a,name='seg_c1_a',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_b = layers.conv2d_layer(ip_layer=seg_c1_a,name='seg_c1_b', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        #Final output layer - Logits before softmax
        seg_fin_layer = layers.conv2d_layer(ip_layer=seg_c1_b,name='seg_fin_layer', num_filters=self.num_classes,use_bias=False, use_relu=False, use_batch_norm=False, training_phase=train_phase)
        actual_cost = loss.dice_loss_with_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)

        # Predict Class
        y_pred = tf.nn.softmax(seg_fin_layer)
        y_pred_cls = tf.argmax(y_pred,axis=3)

        ########################
        # Segmentation loss between predicted labels and true labels
        if(dsc_loss==1):
            # For dice loss function
            # Dice loss without background
            seg_cost = loss.dice_loss_without_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
        elif(dsc_loss==2):
            # Dice loss with background
            seg_cost = loss.dice_loss_with_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
        else:
            # For Weighted Cross Entropy (wCE) loss with background
            seg_cost = loss.pixel_wise_cross_entropy_loss_weighted(logits=seg_fin_layer, labels=y_l_onehot, class_weights=class_weights)
        ########################

        # var list of u-net (segmentation net)
        all_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'enc_' in var_name: all_net_vars.append(v)
            elif 'dec_' in var_name: all_net_vars.append(v)
            elif 'seg_' in var_name: all_net_vars.append(v)

        dec_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'dec_' in var_name: dec_net_vars.append(v)
            if 'seg_' in var_name: dec_net_vars.append(v)

        seg_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'seg_' in var_name: seg_net_vars.append(v)
        
        #print('all_net_var',all_net_vars)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            seg_cost=tf.reduce_mean(seg_cost)

            optimizer_unet_seg = tf.train.AdamOptimizer(learning_rate=learn_rate_seg).minimize(seg_cost,var_list=seg_net_vars)
            optimizer_unet_dec = tf.train.AdamOptimizer(learning_rate=learn_rate_seg).minimize(seg_cost,var_list=dec_net_vars)
            optimizer_unet_all = tf.train.AdamOptimizer(learning_rate=learn_rate_seg).minimize(seg_cost,var_list=all_net_vars)

        seg_summary = tf.summary.scalar('seg_cost', tf.reduce_mean(seg_cost))
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        train_summary = tf.summary.merge([seg_summary])
        # For dice score summary

        mean_dice = tf.placeholder(tf.float32, shape=[], name='mean_dice')
        mean_dice_summary = tf.summary.scalar('mean_val_dice', mean_dice)

        val_totalc = tf.placeholder(tf.float32, shape=[], name='val_totalc')
        val_totalc_sum= tf.summary.scalar('val_totalc_', val_totalc)
        val_summary = tf.summary.merge([mean_dice_summary,val_totalc_sum])

        if(mtask_en==1):
            return {'x': x, 'y_l':y_l, 'train_phase':train_phase, 'seg_cost': seg_cost,'optimizer_unet_seg':optimizer_unet_seg, \
                'y_pred' : y_pred, 'y_pred_cls': y_pred_cls, 'optimizer_unet_dec':optimizer_unet_dec,'actual_cost':actual_cost,\
                'train_summary':train_summary,'seg_fin_layer':seg_fin_layer,'optimizer_unet_all':optimizer_unet_all, \
                'mean_dice':mean_dice,'val_totalc':val_totalc,'val_summary':val_summary, 'dec_c1_a':dec_c1_a}
        else:
            return {'x': x, 'y_l':y_l, 'train_phase':train_phase,'seg_cost': seg_cost,'optimizer_unet_seg':optimizer_unet_seg, \
                'y_pred' : y_pred, 'y_pred_cls': y_pred_cls, 'optimizer_unet_dec':optimizer_unet_dec, 'dec_c1_a':dec_c1_a,\
                'train_summary':train_summary,'seg_fin_layer':seg_fin_layer,'optimizer_unet_all':optimizer_unet_all,\
                'actual_cost':actual_cost,'mean_dice':mean_dice,'val_totalc':val_totalc,'val_summary':val_summary}


    def joint_tr_unet_seg_loss_and_local_cont_loss(self,learn_rate_seg=0.001,dsc_loss=2,en_1hot=1,mtask_en=0,fs_de=2,\
        temp_fac=0.1,batch_size_ft=10,no_of_pos_eles=3,no_of_neg_eles=3,local_loss_exp_no=0,lamda_local=0.1,no_of_cont_classes=3,inf=0):
        # Define the Common encoder-decoder network plus (i) segmentation specific layers to segment the input image, \
        # and (ii) contrastive loss specific layers to compute the contrastive loss

        # No of channels in each level of encoder / decoder
        no_filters=[1, 16, 32, 64, 128, 128]

        if(self.num_classes==2):
            class_weights = tf.constant([[0.1, 0.9]],name='class_weights')
        elif(self.num_classes==3):
            class_weights = tf.constant([[0.1, 0.45, 0.45]],name='class_weights')
        elif(self.num_classes==4):
            class_weights = tf.constant([[0.1, 0.3, 0.3, 0.3]],name='class_weights')
        elif (self.num_classes==8):
            class_weights = tf.constant([[0.09, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13]], name='class_weights')

        num_channels=self.num_channels

        # placeholders for the network
        # Inputs
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y, num_channels], name='x')
        if(en_1hot==1):
            y_l = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y,self.num_classes], name='y_l')
        else:
            y_l = tf.placeholder(tf.int32, shape=[None, self.img_size_x, self.img_size_y], name='y_l')
        train_phase = tf.placeholder(tf.bool, name='train_phase')

        #div_factor = np.power(2,5)//np.power(2,no_of_decoder_blocks)
        div_factor = np.power(2, 5) // np.power(2, 5)
        num_channels=self.num_channels
        num_classes=self.num_classes
        no_of_local_regions=no_of_neg_eles*(num_classes-2)

        pos_indx = tf.placeholder(tf.int32, shape=[None,num_classes-1,no_of_pos_eles,2], name='pos_indx')
        neg_indx = tf.placeholder(tf.int32, shape=[None,num_classes-1,no_of_local_regions,2], name='neg_indx')

        #pos_cls_list = tf.placeholder(tf.int32, shape=[2*batch_size_ft,no_of_cont_classes], name='pos_cls_list')
        
        #if(pix_avg_en!=0):
        y_l_reg = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y,self.num_classes], name='y_l_reg')

        if(en_1hot==0):
            y_l_onehot=tf.one_hot(y_l,depth=self.num_classes)
        else:
            y_l_onehot=y_l
        #print('x,y_l_onehot',x,y_l_onehot)

        ###################################
        # Common Encoder network
        ###################################
        # Last layer from Encoder network
        enc_c6_b,enc_layers_list = self.encoder_network(x, train_phase, no_filters,encoder_list_return=1)

        ###################################
        # skip-connection layers from encoder
        #enc_c1_b,enc_c2_b,enc_c3_b,enc_c4_b,enc_c5_b = enc_layers_list[0],enc_layers_list[1],enc_layers_list[2],enc_layers_list[3],enc_layers_list[4]

        ###################################
        # Common Decoder Network
        ###################################
        dec_c1_a = self.decoder_network(enc_c6_b, train_phase, no_filters, enc_layers_list, fs_de)

        ###################################
        # Segmentation specific layers
        ###################################
        # g_\xi - small network with few convolutions for segmentation loss computation
        seg_c1_a = layers.conv2d_layer(ip_layer=dec_c1_a,name='seg_c1_a',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_b = layers.conv2d_layer(ip_layer=seg_c1_a,name='seg_c1_b', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        #Select which images to pass based on train phase: True - both labeled + unlabeled images; False - only unlabeled images
        lbl_indices = np.arange(0, self.mtask_bs, dtype=np.int32)
        unl_indices = np.arange(int(self.mtask_bs),int(self.mtask_bs)+2*batch_size_ft, dtype=np.int32)
        print('lbl, unl indices',lbl_indices,unl_indices)

        y_l_onehot_reg = y_l_reg
        y_l_onehot=tf.cond(train_phase, lambda:tf.gather(y_l_onehot,lbl_indices), lambda: y_l_onehot)

        seg_c1_b = tf.cond(train_phase, lambda: tf.gather(seg_c1_b,lbl_indices), lambda: seg_c1_b)
        #print('seg_c1_b', seg_c1_b)
        #Final output layer - Logits before softmax
        seg_fin_layer = layers.conv2d_layer(ip_layer=seg_c1_b,name='seg_fin_layer', num_filters=self.num_classes,use_bias=False, use_relu=False, use_batch_norm=False, training_phase=train_phase)
        actual_cost = loss.dice_loss_with_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)

        # Predict Class
        y_pred = tf.nn.softmax(seg_fin_layer)
        y_pred_cls = tf.argmax(y_pred,axis=3)

        ########################
        # Segmentation loss between predicted labels and true labels
        if(dsc_loss==1):
            # For dice loss function
            # Dice loss without background
            seg_cost = loss.dice_loss_without_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
        elif(dsc_loss==2):
            # Dice loss with background
            seg_cost = loss.dice_loss_with_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
        else:
            # For Weighted Cross Entropy (wCE) loss with background
            seg_cost = loss.pixel_wise_cross_entropy_loss_weighted(logits=seg_fin_layer, labels=y_l_onehot, class_weights=class_weights)
        ########################

        ###################################
        # Contrastive loss specific layers
        ###################################
        # Output of common encoder-decoder network - to be feed to a contrastive specific layers
        tmp_dec_layer=dec_c1_a
        tmp_no_filters=no_filters[1]

        # h_\phi - small network with two 1x1 convolutions for contrastive loss computation
        tmp_dec_layer = tf.cond(train_phase, lambda: tf.gather(tmp_dec_layer, unl_indices), lambda: tmp_dec_layer)
        #print('tmp_dec_layer', tmp_dec_layer)
        cont_c1_a = layers.conv2d_layer(ip_layer=tmp_dec_layer,name='cont_c1_a', kernel_size=(1,1), num_filters=tmp_no_filters,use_bias=False, use_relu=True, use_batch_norm=True, training_phase=train_phase)
        cont_c1_b = layers.conv2d_layer(ip_layer=cont_c1_a, name='cont_c1_b', kernel_size=(1,1),num_filters=tmp_no_filters, use_bias=False, use_relu=False,use_batch_norm=False, training_phase=train_phase)

        y_fin_tmp=cont_c1_b
 
        # Define Local Contrastive loss 
        if(inf==1):
            y_fin=y_fin_tmp
            local_loss=1
            net_local_loss=1#np.array(1, dtype=np.int32)
            bs,tmp_batch_size=20,10
            #bs=2*self.batch_size
        else:
            y_fin=y_fin_tmp
            print('y_fin_local',y_fin,pos_indx,neg_indx)
            print('start of for loop',time.ctime())

            local_loss=0
            net_local_loss=0

            for pos_index in range(0,batch_size_ft,1):
            
                index_pos1=pos_index
                index_pos2=batch_size_ft+pos_index
            
                #indexes of positive pair of samples (f(x),f(x')) of input images (x,x') from the batch of feature maps.
                num_i1=np.arange(index_pos1,index_pos1+1,dtype=np.int32)
                num_i2=np.arange(index_pos2,index_pos2+1,dtype=np.int32)
            
                # gather required positive samples (f(x),f(x')) of (x,x') for the numerator term
                x_num_i1=tf.gather(y_fin,num_i1)
                x_num_i2=tf.gather(y_fin,num_i2)
                #print('x_num_i1',index_pos1,index_pos2,x_num_i1,x_num_i2)

                x_zero_re=tf.constant(0,shape=(),dtype=np.float32)
                x_zero_vec=tf.constant(0,dtype=np.float32,shape=(1,tmp_no_filters))

                #x_epsilon_vec=tf.constant(0.0,dtype=np.float32,shape=(1,tmp_no_filters))
                #print('x_zero_vec',x_zero_vec)

                mask_i1 = tf.gather(y_l_reg, num_i1)
                mask_i2 = tf.gather(y_l_reg, num_i2)

                if(self.dataset_name=='mmwhs'):
                    #pos_cls_ref = np.asarray([0,2,3,4,5,6])
                    #neg_cls_ref = np.asarray([1,3,4,5,6,7])
                    #pos_cls_ref = np.asarray([0,1,2,3,5,6])
                    #neg_cls_ref = np.asarray([1,2,3,4,6,7])
                    pos_cls_ref = np.asarray([0,1,2,3,4,5,6])
                    neg_cls_ref = np.asarray([1,2,3,4,5,6,7])
                    print('pos_cls_ref,neg_cls_ref',pos_cls_ref,neg_cls_ref)
                elif(self.dataset_name=='acdc'):
                    pos_cls_ref = np.asarray([0,1,2])
                    neg_cls_ref = np.asarray([1,2,3])
                    print('pos_cls_ref,neg_cls_ref', pos_cls_ref, neg_cls_ref)
                elif(self.dataset_name=='prostate_md'):
                    pos_cls_ref = np.asarray([0,1])
                    neg_cls_ref = np.asarray([1,2])
                    print('pos_cls_ref,neg_cls_ref', pos_cls_ref, neg_cls_ref)

                for pos_cls in pos_cls_ref:
                    pos_cls_ele_i1=pos_indx[pos_index][pos_cls]
                    neg_cls_ele_i1=neg_indx[pos_index][pos_cls]
                    pos_cls_ele_i2=pos_indx[batch_size_ft+pos_index][pos_cls]
                    neg_cls_ele_i2=neg_indx[batch_size_ft+pos_index][pos_cls]

                    #print('cls_i1',pos_cls,pos_cls_ele_i1,neg_cls_ele_i1)
                    #print('cls_i2',pos_cls,pos_cls_ele_i2,neg_cls_ele_i2)

                    #############################
                    #mask of image 1 (x) from batch X_B
                    #select positive classes masks' mean embeddings
                    mask_i1_pos=tf.gather(mask_i1,pos_cls+1,axis=-1)
                    ##mask_i1_pos=tf.gather(mask_i1,pos_cls,axis=-1)
                    pos_cls_avg_i1 = tf.boolean_mask(x_num_i1, mask_i1_pos)
                    pos_avg_vec_i1_p = tf.reshape(tf.reduce_mean(pos_cls_avg_i1,axis=0),(1,tmp_no_filters))
                    pos_avg_i1_nan=tf.is_nan(tf.reduce_sum(pos_avg_vec_i1_p))
                    ##print('pos_avg_vec_i1',pos_cls,mask_i1_pos,pos_cls_avg_i1,pos_avg_vec_i1_p)
                    #############################

                    #############################
                    # make list of negative classes masks' mean embeddings from image 1 (x) mask
                    neg_mask1_list = []
                    for neg_cls_i1 in neg_cls_ref:
                        mask_i1_neg = tf.gather(mask_i1, neg_cls_i1, axis=-1)
                        neg_cls_avg_i1 = tf.boolean_mask(x_num_i1, mask_i1_neg)
                        neg_avg_vec_i1_p = tf.reshape(tf.reduce_mean(neg_cls_avg_i1, axis=0), (1, tmp_no_filters))
                        neg_avg_i1_nan=tf.is_nan(tf.reduce_sum(neg_avg_vec_i1_p))
                        neg_mask1_list.append(neg_avg_vec_i1_p)
                    #print('neg_mask1_list', neg_mask1_list)
                    #############################

                    #############################
                    #mask of image 2 (x')  from batch X_B
                    #select positive classes masks' mean embeddings
                    mask_i2_pos=tf.gather(mask_i2,pos_cls+1,axis=-1)
                    ##mask_i2_pos=tf.gather(mask_i2,pos_cls,axis=-1)
                    pos_cls_avg_i2 = tf.boolean_mask(x_num_i2, mask_i2_pos)
                    pos_avg_vec_i2_p = tf.reshape(tf.reduce_mean(pos_cls_avg_i2,axis=0),(1,tmp_no_filters))
                    pos_avg_i2_nan=tf.is_nan(tf.reduce_sum(pos_avg_vec_i2_p))
                    #print('pos_avg_vec_i2',pos_cls,mask_i2_pos,pos_cls_avg_i2,pos_avg_vec_i2_p)
                    #############################

                    #############################
                    # #select negative classes mask averages
                    # make list of negative classes masks' mean embeddings from image 2 (x') mask
                    neg_mask2_list = []
                    for neg_cls_i2 in neg_cls_ref:
                        mask_i2_neg = tf.gather(mask_i2, neg_cls_i2, axis=-1)
                        neg_cls_avg_i2 = tf.boolean_mask(x_num_i2, mask_i2_neg)
                        neg_avg_vec_i2_p = tf.reshape(tf.reduce_mean(neg_cls_avg_i2, axis=0), (1, tmp_no_filters))
                        neg_avg_i2_nan=tf.is_nan(tf.reduce_sum(neg_avg_vec_i2_p))
                        neg_mask2_list.append(neg_avg_vec_i2_p)
                    #print('neg_mask2_list', neg_mask2_list)
                    #############################

                    #Loop over all the positive embeddings from f(x) of all classes
                    for n_pos_idx in range(0,no_of_pos_eles,1):
                        x_num_tmp_i1 = tf.gather(x_num_i1,pos_cls_ele_i1[n_pos_idx][0],axis=1)
                        #print('x_num_tmp_i1 j0',x_num_tmp_i1)
                        x_num_tmp_i1 = tf.gather(x_num_tmp_i1,pos_cls_ele_i1[n_pos_idx][1],axis=1)
                        #print('x_num_tmp_i1 j1',x_num_tmp_i1)

                        x_n1_count=tf.math.count_nonzero(x_num_tmp_i1)
                        x_n_i1_flat = tf.cond(tf.equal(x_n1_count,0), lambda: x_zero_vec, lambda: tf.layers.flatten(inputs=x_num_tmp_i1))
                        #print('x_n_i1_flat',x_n_i1_flat,tf.layers.flatten(inputs=x_num_tmp_i1))
                        x_w3_n_i1=x_n_i1_flat

                        x_num_tmp_i2 = tf.gather(x_num_i2,pos_cls_ele_i2[n_pos_idx][0],axis=1)
                        x_num_tmp_i2 = tf.gather(x_num_tmp_i2,pos_cls_ele_i2[n_pos_idx][1],axis=1)
                        #print('x_num_tmp_i2 j',x_num_tmp_i2)

                        x_n2_count=tf.math.count_nonzero(x_num_tmp_i2)
                        x_n_i2_flat = tf.cond(tf.equal(x_n2_count,0), lambda: x_zero_vec, lambda: tf.layers.flatten(inputs=x_num_tmp_i2))
                        #print('x_n_i2_flat',x_n_i2_flat,tf.layers.flatten(inputs=x_num_tmp_i2))
                        x_w3_n_i2=x_n_i2_flat

                        # Cosine loss for positive pair of pixel embeddings from f(x), f(x')
                        # Numerator loss terms of local loss
                        pos_avg_vec_i1=pos_avg_vec_i1_p
                        pos_avg_vec_i2=pos_avg_vec_i2_p

                        log_or_n1 = tf.math.logical_or(tf.equal(x_n1_count,0),tf.equal(tf.math.count_nonzero(pos_avg_vec_i1),0))
                        log_or_n1_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i1)),tf.is_nan(tf.reduce_sum(pos_avg_vec_i1)))
                        log_or_n1_net = tf.math.logical_or(log_or_n1,log_or_n1_nan)
                        num_i1_ss = tf.cond(log_or_n1_net, lambda: x_zero_re, lambda: self.cos_sim(x_w3_n_i1,pos_avg_vec_i1,temp_fac))
                        log_or_n2 = tf.math.logical_or(tf.equal(x_n2_count,0),tf.equal(tf.math.count_nonzero(pos_avg_vec_i2),0))
                        log_or_n2_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i2)),tf.is_nan(tf.reduce_sum(pos_avg_vec_i2)))
                        log_or_n2_net = tf.math.logical_or(log_or_n2, log_or_n2_nan)
                        num_i2_ss = tf.cond(log_or_n2_net, lambda: x_zero_re, lambda: self.cos_sim(x_w3_n_i2,pos_avg_vec_i2,temp_fac))

                        if(local_loss_exp_no==1):
                            log_or_i1_i2 = tf.math.logical_or(tf.equal(x_n1_count,0),tf.equal(tf.math.count_nonzero(pos_avg_vec_i2),0))
                            log_or_i1_i2_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i1)),tf.is_nan(tf.reduce_sum(pos_avg_vec_i2)))
                            log_or_i1_i2_net = tf.math.logical_or(log_or_i1_i2, log_or_i1_i2_nan)
                            num_i1_i2_ss = tf.cond(log_or_i1_i2_net, lambda: x_zero_re, lambda: self.cos_sim(x_w3_n_i1,pos_avg_vec_i2,temp_fac))

                            log_or_i2_i1 = tf.math.logical_or(tf.equal(x_n2_count,0),tf.equal(tf.math.count_nonzero(pos_avg_vec_i1),0))
                            log_or_i2_i1_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i2)),tf.is_nan(tf.reduce_sum(pos_avg_vec_i1)))
                            log_or_i2_i1_net = tf.math.logical_or(log_or_i2_i1,log_or_i2_i1_nan)
                            num_i2_i1_ss = tf.cond(log_or_i2_i1_net, lambda: x_zero_re, lambda: self.cos_sim(x_w3_n_i2,pos_avg_vec_i1,temp_fac))

                        # Denominator loss terms of local loss
                        den_i1_ss,den_i2_ss=0,0
                        den_i1_i2_ss,den_i2_i1_ss=0,0

                        #############################
                        # compute loss for positive mean class pixels from mask of image 1 (x)
                        # negatives from mask of image 1 (x)
                        for neg_avg_i1_c1 in neg_mask1_list:
                            log_or_n1_d1 = tf.math.logical_or(tf.equal(x_n1_count, 0),tf.equal(tf.math.count_nonzero(neg_avg_i1_c1), 0))
                            log_or_n1_d1_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i1)),tf.is_nan(tf.reduce_sum(neg_avg_i1_c1)))
                            log_or_n1_d1_net = tf.math.logical_or(log_or_n1_d1,log_or_n1_d1_nan)
                            den_i1_ss = den_i1_ss + tf.cond(log_or_n1_d1_net, lambda: x_zero_re,lambda: tf.exp(self.cos_sim(x_w3_n_i1, neg_avg_i1_c1, temp_fac)))

                        # negatives from mask of image 2 (x')
                        if (local_loss_exp_no == 1):
                            for neg_avg_i1_c2 in neg_mask2_list:
                                log_or_n1_d2 = tf.math.logical_or(tf.equal(x_n1_count, 0),tf.equal(tf.math.count_nonzero(neg_avg_i1_c2), 0))
                                log_or_n1_d2_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i1)),tf.is_nan(tf.reduce_sum(neg_avg_i1_c2)))
                                log_or_n1_d2_net = tf.math.logical_or(log_or_n1_d2,log_or_n1_d2_nan)
                                den_i1_ss = den_i1_ss + tf.cond(log_or_n1_d2_net, lambda: x_zero_re,lambda: tf.exp(self.cos_sim(x_w3_n_i1, neg_avg_i1_c2, temp_fac)))
                        #############################

                        #############################
                        # compute loss for positive avg class pixels from mask of image 2 (x')
                        # negatives from mask of image 2 (x')
                        for neg_avg_i2_c2 in neg_mask2_list:
                            log_or_n2_d2 = tf.math.logical_or(tf.equal(x_n2_count, 0),tf.equal(tf.math.count_nonzero(neg_avg_i2_c2),0))
                            log_or_n2_d2_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i2)),tf.is_nan(tf.reduce_sum(neg_avg_i2_c2)))
                            log_or_n2_d2_net = tf.math.logical_or(log_or_n2_d2,log_or_n2_d2_nan)
                            den_i2_ss = den_i2_ss + tf.cond(log_or_n2_d2_net, lambda: x_zero_re,lambda: tf.exp(self.cos_sim(x_w3_n_i2, neg_avg_i2_c2,temp_fac)))

                        # negatives from mask of image 1 (x)
                        if (local_loss_exp_no == 1):
                            for neg_avg_i2_c1 in neg_mask1_list:
                                log_or_n2_d1 = tf.math.logical_or(tf.equal(x_n2_count, 0), tf.equal(tf.math.count_nonzero(neg_avg_i2_c1), 0))
                                log_or_n2_d1_nan = tf.math.logical_or(tf.is_nan(tf.reduce_sum(x_w3_n_i2)),tf.is_nan(tf.reduce_sum(neg_avg_i2_c1)))
                                log_or_n2_d1_net = tf.math.logical_or(log_or_n2_d1,log_or_n2_d1_nan)
                                den_i2_ss = den_i2_ss + tf.cond(log_or_n2_d1_net, lambda: x_zero_re,lambda: tf.exp(self.cos_sim(x_w3_n_i2, neg_avg_i2_c1,temp_fac)))

                        log_num_i1_nan = tf.math.logical_or(tf.is_nan(tf.exp(num_i1_ss)),tf.is_nan(tf.exp(den_i1_ss)))
                        log_num_i1_nan = tf.squeeze(log_num_i1_nan)
                        log_num_i1_zero = tf.math.logical_or(tf.equal(tf.math.count_nonzero(num_i1_ss), 0),tf.equal(tf.math.count_nonzero(den_i1_ss), 0))
                        log_num_i1_net = tf.math.logical_or(log_num_i1_zero, log_num_i1_nan)
                        num_i1_loss = tf.cond(log_num_i1_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i1_ss))/(tf.math.reduce_sum(den_i1_ss))))
                        local_loss = local_loss + num_i1_loss

                        log_num_i2_nan = tf.math.logical_or(tf.is_nan(tf.exp(num_i2_ss)),tf.is_nan(tf.exp(den_i2_ss)))
                        log_num_i2_nan = tf.squeeze(log_num_i2_nan)
                        log_num_i2_zero = tf.math.logical_or(tf.equal(tf.math.count_nonzero(num_i2_ss), 0),tf.equal(tf.math.count_nonzero(den_i2_ss), 0))
                        log_num_i2_net = tf.math.logical_or(log_num_i2_zero, log_num_i2_nan)
                        num_i2_loss = tf.cond(log_num_i2_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i2_ss))/(tf.math.reduce_sum(den_i2_ss))))
                        local_loss = local_loss + num_i2_loss
                        if (local_loss_exp_no == 1):
                            #local loss from feature map f(x) of image 1 (x)
                            #log_num_i1_i2 = tf.math.logical_or(tf.equal(tf.math.reduce_sum(tf.exp(num_i1_i2_ss)),0),tf.equal((tf.math.reduce_sum(tf.exp(num_i1_i2_ss))+tf.math.reduce_sum(den_i1_ss)),0))
                            log_num_i1_i2_nan = tf.math.logical_or(tf.is_nan(tf.exp(num_i1_i2_ss)),tf.is_nan(tf.exp(den_i1_ss)))
                            log_num_i1_i2_nan = tf.squeeze(log_num_i1_i2_nan)
                            log_num_i1_i2_zero = tf.math.logical_or(tf.equal(tf.math.count_nonzero(num_i1_i2_ss), 0),tf.equal(tf.math.count_nonzero(den_i1_ss), 0))
                            log_num_i1_i2_net = tf.math.logical_or(log_num_i1_i2_zero, log_num_i1_i2_nan)
                            #local_loss = local_loss + tf.cond(log_num_i1_i2_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i1_i2_ss))/(tf.math.reduce_sum(tf.exp(num_i1_i2_ss))+tf.math.reduce_sum(den_i1_ss))))
                            local_loss = local_loss + tf.cond(log_num_i1_i2_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i1_i2_ss))/(tf.math.reduce_sum(den_i1_ss))))

                            # local loss from feature map f(x') of image 2 (x')
                            #log_num_i2_i1 = tf.math.logical_or(tf.equal(tf.math.reduce_sum(tf.exp(num_i2_i1_ss)),0),tf.equal((tf.math.reduce_sum(tf.exp(num_i2_i1_ss))+tf.math.reduce_sum(den_i2_ss)),0))
                            log_num_i2_i1_nan = tf.math.logical_or(tf.is_nan(tf.exp(num_i2_i1_ss)),tf.is_nan(tf.exp(den_i2_ss)))
                            log_num_i2_i1_nan = tf.squeeze(log_num_i2_i1_nan)
                            log_num_i2_i1_zero = tf.math.logical_or(tf.equal(tf.math.count_nonzero(num_i2_i1_ss), 0),tf.equal(tf.math.count_nonzero(den_i2_ss), 0))
                            log_num_i2_i1_net = tf.math.logical_or(log_num_i2_i1_zero,log_num_i2_i1_nan)
                            #local_loss = local_loss + tf.cond(log_num_i2_i1_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i2_i1_ss))/(tf.math.reduce_sum(tf.exp(num_i2_i1_ss))+tf.math.reduce_sum(den_i2_ss))))
                            local_loss = local_loss + tf.cond(log_num_i2_i1_net, lambda: x_zero_re, lambda: -tf.log(tf.math.reduce_sum(tf.exp(num_i2_i1_ss))/(tf.math.reduce_sum(den_i2_ss))))

                if (local_loss_exp_no == 1):
                    local_loss=local_loss/(2*no_of_pos_eles*(num_classes-1))
                else:
                    local_loss=local_loss/(no_of_pos_eles*(num_classes-1))

            net_local_loss=local_loss/batch_size_ft

            
        print('end of for loop',time.ctime())
 
        # var list of u-net (segmentation net)
        all_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'enc_' in var_name: all_net_vars.append(v)
            elif 'dec_' in var_name: all_net_vars.append(v)
            elif 'seg_' in var_name: all_net_vars.append(v)
            elif 'cont_' in var_name: all_net_vars.append(v)

        dec_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'dec_' in var_name: dec_net_vars.append(v)
            if 'seg_' in var_name: dec_net_vars.append(v)
            if 'cont_' in var_name: dec_net_vars.append(v)

        seg_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'seg_' in var_name: seg_net_vars.append(v)

        if(inf!=1):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                cont_loss_cost=lamda_local * tf.reduce_mean(net_local_loss)
                seg_cost=tf.reduce_mean(seg_cost)
                net_cost= seg_cost + cont_loss_cost

                optimizer_unet_seg = tf.train.AdamOptimizer(learning_rate=learn_rate_seg).minimize(net_cost,var_list=seg_net_vars)
                optimizer_unet_dec = tf.train.AdamOptimizer(learning_rate=learn_rate_seg).minimize(net_cost,var_list=dec_net_vars)
                optimizer_unet_all = tf.train.AdamOptimizer(learning_rate=learn_rate_seg).minimize(net_cost,var_list=all_net_vars)

            seg_summary = tf.summary.scalar('seg_cost', tf.reduce_mean(seg_cost))
            cont_summary = tf.summary.scalar('cont_loss_cost', tf.reduce_mean(cont_loss_cost))
            net_summary = tf.summary.scalar('net_cost', tf.reduce_mean(net_cost))
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            #train_summary = tf.summary.merge([seg_summary,cont_summary,net_summary])
            train_summary = tf.summary.merge([seg_summary])
            # For dice score summary

        mean_dice = tf.placeholder(tf.float32, shape=[], name='mean_dice')
        mean_dice_summary = tf.summary.scalar('mean_val_dice', mean_dice)

        val_totalc = tf.placeholder(tf.float32, shape=[], name='val_totalc')
        val_totalc_sum= tf.summary.scalar('val_totalc_', val_totalc)
        val_summary = tf.summary.merge([mean_dice_summary,val_totalc_sum])


        if(inf==1):
            return {'x':x, 'y_l':y_l,'y_l_reg':y_l_reg,'pos_indx':pos_indx, 'neg_indx':neg_indx,'train_phase':train_phase,'y_pred':y_pred,\
                'actual_cost':actual_cost,'y_pred_cls': y_pred_cls, 'dec_c1_a':dec_c1_a, 'y_fin_tmp':y_fin_tmp}
        else:
            return {'x':x, 'y_l':y_l,'y_l_reg':y_l_reg, 'pos_indx':pos_indx, 'neg_indx':neg_indx,'train_phase':train_phase, 'net_cost':net_cost, \
                'seg_cost': seg_cost,'cont_loss_cost':cont_loss_cost,'actual_cost':actual_cost,'y_pred_cls': y_pred_cls,\
                'optimizer_unet_dec':optimizer_unet_dec,'optimizer_unet_all':optimizer_unet_all,\
                'x_num_tmp_i1':x_num_tmp_i1,'x_num_tmp_i2':x_num_tmp_i2,\
                'x_n1_count':x_n1_count,'x_n2_count':x_n2_count,\
                'num_i1_ss':num_i1_ss,'num_i2_ss':num_i2_ss,'den_i1_ss':den_i1_ss,'den_i2_ss':den_i2_ss,\
                'num_i1_loss':num_i1_loss,'num_i2_loss':num_i2_loss,'mean_dice':mean_dice, 'dec_c1_a':dec_c1_a, 'y_fin_tmp':y_fin_tmp,\
                'train_summary':train_summary, 'y_pred':y_pred,'val_totalc':val_totalc, 'val_summary':val_summary}
