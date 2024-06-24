import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import cv2
import pdb

class InteractiveSegmenter:
    def __init__(self):
        device='cuda'
        self.current_dir = os.path.dirname(__file__)
        sam_checkpoint = os.path.join(self.current_dir, "weights", "sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.coordinates = []
        
    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def on_click(self, event, ax, fig):
        if event.inaxes is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.coordinates.append((x, y))
            ax.plot(x, y, 'ro')  # Mark the clicked points with red circles
            fig.canvas.draw()
            print(f"Recorded coordinate: (x={x}, y={y})")

    def interactive_coordinate_recorder(self, image, size, obj_name):
        self.coordinates = []  # Reset the coordinates
        fig, ax = plt.subplots()

        # Load and resize the image
        original_image = Image.fromarray(image)
        resized_image = original_image.resize(size, resample=Image.NEAREST)  # Resize the image
        image = np.array(resized_image)  # Convert the PIL image to a numpy array

        ax.imshow(image)
        # Connect the click event to the on_click function
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click(event, ax, fig))
        plt.title(f'Click on the image to record points for {obj_name}')
        plt.show()

    def segment_image(self, image, obj_name):        
        # Interactive coordinate recorder
        size = (image.shape[1], image.shape[0])
        self.interactive_coordinate_recorder(image, size, obj_name)
        coordinates = np.array(self.coordinates)
        print("Coordinates:", coordinates)

        self.predictor.set_image(image)

        input_label = np.ones(len(coordinates))
        masks, _, _ = self.predictor.predict(
            point_coords=coordinates,
            point_labels=input_label,
            multimask_output=False,
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        self.show_mask(masks, plt.gca())
        self.show_points(coordinates, input_label, plt.gca())
        plt.axis('off')
        plt.show()

        return masks

# Usage example:


if __name__ == '__main__':

    segmenter = InteractiveSegmenter()

    # Load the image
    image = cv2.imread('./demo_data/cup/images/00001.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    binary_mask = segmenter.segment_image(image)

    # show the binary mask
    plt.imshow(binary_mask[0])
    plt.axis('off')
    plt.show()
