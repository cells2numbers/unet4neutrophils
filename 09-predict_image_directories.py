
# coding: utf-8

# # Last step
# # Predict segmentations

# In[1]:




# In[2]:


import os
import os.path

import glob

import matplotlib.pyplot as plt
import numpy as np

import skimage.io
import skimage.morphology

import tqdm

import tensorflow as tf
import keras

import utils.metrics
import utils.model_builder


# Configurable parameter:
# * config (experiment)
# * GPU
# * input folder 
# * output folder

# # Configuration

# In[3]:


GPU_NO = "1"
input_directory = "/storage/data/2018_tim_tracking/2018_06_29_lineage_tracking_blainey/images/movies/12_2018_MCF10_Drug_full/images"
# "/storage/data/2018_tim_tracking/2018_08_06_migration_perturbations_asthma/images/2018_asthma_migration_data_kira"
output_directory = "/storage/data/2018_tim_tracking/2018_06_29_lineage_tracking_blainey/images/movies/12_2018_MCF10_Drug_full/predictions"
#"/storage/data/2018_tim_tracking/2018_08_06_migration_perturbations_asthma/images/2018_asthma_migration_data_kira_predictions"
experiment_name = "09"


# # Initialize keras 

# In[4]:


from config import config_vars

# load configuration file
config_vars = utils.dirtools.setup_experiment(config_vars, experiment_name)

# output directory
config_vars["probmap_out_dir"] = output_directory

# initialize GPU
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = GPU_NO

session = tf.Session(config = configuration)

# apply session
keras.backend.set_session(session)



# # Load and preprocessing images

# In[5]:


def load_and_prepare_images_from_directory(input_directory):
    image_names = glob.glob(input_directory + "/*png")
    images = []
    imagebuffer = skimage.io.imread_collection( image_names )
    for image in imagebuffer:
        #image = skimage.io.imread(filename)
        images.append(skimage.color.rgb2gray(image))

    #print("found {} images".format(len(imagebuffer)))

    images = np.array(images)

    dim1 = np.floor(images.shape[1]/16) * 16 
    dim1 = dim1.astype(np.int)
    dim2 = np.floor(images.shape[2]/16) * 16 
    dim2 = dim2.astype(np.int)

    images = images[:,0:dim1,0:dim2]

    dim1 = images.shape[1]
    dim2 = images.shape[2]

    images = images.reshape((-1, dim1, dim2, 1))

    #print(dim1,dim2)

    # preprocess images
    percentile = 99.9
    for image_no in range(images.shape[0]):
        orig_img = images[image_no,:,:,:]

        high = np.percentile(orig_img, percentile)
        low = np.percentile(orig_img, 100-percentile)

        img = np.minimum(high, orig_img)
        img = np.maximum(low, img)
        img = (img - low) / (high - low) 
        img = skimage.img_as_ubyte(img) 
        images[image_no,:,:,:] = img # gives float64, thus cast to 8 bit later



    images = images.astype(float)
    images = images / 256

    return(images,imagebuffer)


# # Predict images 

# In[6]:


for directory in tqdm.tqdm(os.listdir(input_directory)):
    
    if os.path.exists(os.path.join(output_directory,directory)):
        print("Folder {} processed!".format(directory))
    else:
        print("Predicting images in folder {}".format(directory))  
    
        [images,imagebuffer] = load_and_prepare_images_from_directory(os.path.join(input_directory,directory))

        # build model and load weights
        dim1 = images.shape[1]
        dim2 = images.shape[2]
        model = utils.model_builder.get_model_3_class(dim1, dim2)
        model.load_weights(config_vars["model_file"])

        #  prediction 
        #for i in 
        
        
        predictions = model.predict(images, batch_size=1)


        
        os.makedirs(os.path.join(output_directory,directory)) 
        print("Saving image data into folder {}".format(output_directory))
        for i in range(len(images)):

            image_savename = os.path.join(
                output_directory, 
                directory, 
                os.path.basename(imagebuffer.files[i])
            )

            probmap = predictions[i].squeeze()

            skimage.io.imsave(os.path.splitext(image_savename)[0] + ".png", probmap)

    
    

