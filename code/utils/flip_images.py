import glob
import os
import numpy as np
from scipy import misc

train_frame_path="/home/ubuntu/RoboND-DeepLearning-Project/data/train/images"
validation_frame_path="/home/ubuntu/RoboND-DeepLearning-Project/data/train/masks"

files = glob.glob(os.path.join(train_frame_path,"*.jpeg"))
for i in files:
	frame = misc.imread(i)
	frame_flip = np.fliplr(frame)
	fname = i.replace("cam1","cam1_flip")
	misc.imsave(os.path.join(train_frame_path,fname),frame_flip)

files = glob.glob(os.path.join(validation_frame_path,"*.png"))
for i in files:
	frame = misc.imread(i)
	frame_flip = np.fliplr(frame)
	fname = i.replace("_mask","_mask_flip")
	misc.imsave(os.path.join(validation_frame_path,fname),frame_flip)
