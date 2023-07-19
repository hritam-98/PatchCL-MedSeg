import math
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

def augment(imgs):
    """
    INPUT: numpy array of shape num_samples*3*img_size*img_size [dtype = float32]
    OUTPUT = numpy array of same shape as input with augmented images
    """

    imgs = (imgs*255).astype(np.uint8)
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    aug = iaa.Sequential(
        # Define our sequence of augmentation steps that will be applied to every image.
        [
            iaa.Fliplr(0.7), # horizontally flip 70% of all images
            # iaa.Flipud(0.2), # vertically flip 20% of all images
            
            # crop some of the images by 0-20% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.2))),
            
            iaa.SomeOf((1,5),[
                              # Add gaussian noise to some images.
                              # In 50% of these cases, the noise is randomly sampled per
                              # channel and pixel.
                              # In the other 50% of all cases it is sampled once per
                              # pixel (i.e. brightness change).
                              iaa.AdditiveGaussianNoise(
                                  loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                              ),
                          
                              #color jitter the image == change brightness, contrast, hue and saturation                     
                              iaa.AddToHueAndSaturation((-50, 50), per_channel=0.5),
                              iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
                              iaa.Multiply((0.5, 1.5), per_channel=0.5),

                              # Convert each image to grayscale and then overlay the
                              # result with the original with random alpha. I.e. remove
                              # colors with varying strengths.
            ]),
         sometimes(iaa.Grayscale(alpha=(0.0, 1.0)))
        ]
    )

    imgs = aug(images=imgs)
    imgs = (imgs/255.).astype(np.float32)

    return imgs

def batch_augment(list_imgs, N):
    """
    INPUT: 
    list_imgs =list of (numpy array of shape num_classes*3*img_size*img_size [dtype = float32])
    N = (int) total number of samples in a batch
    OUTPUT = list of numpy array of same shape as input with augmented images
    """
    num_classes_in_batch = 0
    for imgs in list_imgs:
      if imgs is not None:
          num_classes_in_batch+=1
    
    imgs_per_class= int(math.ceil(N/num_classes_in_batch))
    out_list =[]
    for imgs in list_imgs:
        if imgs is not None:
            if imgs.shape[0]>=imgs_per_class:
              np.random.shuffle(imgs)
              imgs = imgs[:imgs_per_class,:,:,:]
              out_list.append(imgs)
            else :
                imgs_in_cls =int(imgs.shape[0])
                num_augs = int(math.ceil(imgs_per_class/imgs_in_cls))-1
                imgs_to_append = imgs
                for i in range(num_augs):
                    np.random.shuffle(imgs)
                    aug_imgs = augment(imgs.transpose(0,2,3,1))
                    imgs_to_append=np.vstack((imgs_to_append, aug_imgs.transpose(0,3,1,2)))
                imgs_to_append = imgs_to_append[:imgs_per_class,:,:,:]
                out_list.append(imgs_to_append)   
        else :
            out_list.append(None)

    return out_list