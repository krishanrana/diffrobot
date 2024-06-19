import os

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

from matplotlib import pyplot as plt

import pdb


@torch.inference_mode()
@torch.cuda.amp.autocast()
def main():
    # obtain the Cutie model with default parameters -- skipping hydra configuration
    cutie = get_default_model()
    # Typically, use one InferenceCore per video
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    image_path = './demo_data/cup/images'
    # ordering is important
    images = sorted(os.listdir(image_path))

    # mask for the first frame
    # NOTE: this should be a grayscale mask or a indexed (with/without palette) mask,
    # and definitely NOT a colored RGB image
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html: mode "L" or "P"
    mask = Image.open('./demo_data/cup/mask.png')
    assert mask.mode in ['L', 'P']

    # palette is for visualization
    # palette = mask.getpalette()

    # the number of objects is determined by counting the unique values in the mask
    # common mistake: if the mask is resized w/ interpolation, there might be new unique values
    objects = np.unique(np.array(mask))
    # background "0" does not count as an object
    objects = objects[objects != 0].tolist()
    mask = torch.from_numpy(np.array(mask)).cuda()

    for ti, image_name in enumerate(images):
        # load the image as RGB; normalization is done within the model
        image = Image.open(os.path.join(image_path, image_name))

        

        resized_image_pil = image.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
        resized_image = torch.from_numpy(np.array(resized_image_pil)).permute(2, 0, 1)

        # resize the image to match the mask shape
        #image = image.resize(mask.size(), Image.BILINEAR)
        image = resized_image.cuda().float()

        # visualize the input image
        plt.imshow(image.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        plt.show()

        if ti == 0:
            # if mask is passed in, it is memorized
            # if not all objects are specified, we propagate the unspecified objects using memory
            output_prob = processor.step(image, mask, objects=objects)
        else:
            # otherwise, we propagate the mask from memory
            output_prob = processor.step(image)

        # convert output probabilities to an object mask
        mask = processor.output_prob_to_mask(output_prob)
        # visualize prediction
        mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
        # mask.putpalette(palette)
        # mask.show()  # or use mask.save(...) to save it somewhere

        # visualize the prediction
        plt.imshow(mask)
        plt.show()

   




main()