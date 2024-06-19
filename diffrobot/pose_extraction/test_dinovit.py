import matplotlib.pyplot as plt
import torch
from correspondences import find_correspondences, draw_correspondences


#@title Configuration:
#@markdown Choose image paths:

image_path1 = 'images/cup_close1.png' #@param
image_path2 = 'images/cup2.png' #@param

#@markdown Choose number of points to output:
num_pairs = 20 #@param
#@markdown Choose loading size:
load_size = 224 #@param
#@markdown Choose layer of descriptor:
layer = 9 #@param
#@markdown Choose facet of descriptor:
facet = 'key' #@param
#@markdown Choose if to use a binned descriptor:
bin=True #@param
#@markdown Choose fg / bg threshold:
thresh=0.05 #@param
#@markdown Choose model type:
model_type='dino_vits8' #@param
#@markdown Choose stride:
stride=4 #@param

# apply a mask to the image to extract the region of interest
# then apply the model to extract the descriptors and find the correspondences


with torch.no_grad():
    points1, points2, image1_pil, image2_pil = find_correspondences(image_path1, image_path2, num_pairs, load_size, layer,
                                                                   facet, bin, thresh, model_type, stride)
fig_1, ax1 = plt.subplots()
ax1.axis('off')
ax1.imshow(image1_pil)
fig_2, ax2 = plt.subplots()
ax2.axis('off')
ax2.imshow(image2_pil)


fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
plt.show()