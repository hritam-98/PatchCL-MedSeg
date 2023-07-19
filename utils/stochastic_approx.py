
import numpy as np




class StochasticApprox():
    def __init__(self,num_classes, init_threshold,target_perc,patch_size=256):
        self.num_classes=num_classes
        self.patch_size=patch_size
        self.init_threshold=init_threshold
        self.target=target_perc
        self.thresholds = np.ones((self.num_classes))*self.init_threshold
        self.time = np.zeros((self.num_classes))
        #self.history=[]
    
    """
    INPUTS:
    p_list: list of len=num_classes of ndarray of shape (num_patches X 3 X patch_size X patch_size)
    init_thresholds: Enter foat in (0,1) to set that initial threshold percentage
    target_perc: Enter target percentage of patches to qualify in (0,1)
    patch_size(int)
    """

    def time_step(self,n):
        if n == 0:
            return 0.01
        else:
            return 0.01/n
    
    def update(self,patch_list):
        for ind in range(len(patch_list)):
            if patch_list[ind] is not None:
                num_patches = patch_list[ind].shape[0]                          #Number of patches incoming for a given class
                sum_vect = np.sum(patch_list[ind],axis=1)                       #summing across all channels
                sum_vect[sum_vect>0]=1                                          #this converts it into a binary img
                sum_vect= np.sum(sum_vect.reshape(num_patches,-1),axis=1)       #no of pixels in each patch for the given class
                bool_vect = sum_vect > (self.thresholds[ind]*self.patch_size**2)#qualifying patches
                false_indices=np.where(bool_vect)[0]
                patch_list[ind]=np.delete(patch_list[ind],false_indices,axis=0) #removing disqualified patches

                #updating threshold
                self.thresholds[ind]=self.thresholds[ind]+self.time_step(self.time[ind])*(self.target*num_patches-patch_list[ind].shape[0])
                self.time[ind]+=1

                #if nothing qualifies, convert to None
                if patch_list[ind].shape[0]==0:
                    patch_list[ind] = None

            self.thresholds[self.thresholds>1]=1
            self.thresholds[self.thresholds<0]=0
        #self.history.append(self.thresholds)
        return patch_list