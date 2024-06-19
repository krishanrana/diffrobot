from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import matplotlib.pyplot as plt
import pdb
import numpy as np
from PIL import Image

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'


class ImageMatcher:
    def __init__(self, device_type='auto', extractor_config={}, matcher_config={}):
        if device_type == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_type)
        
        self.extractor = SuperPoint(max_num_keypoints=5048).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint",
                                 depth_confidence=-1, 
                                 width_confidence=-1,
                                 filter_threshold=0.01).eval().to(self.device)
        self.coordinates = []
        self.fig = None
        self.ax = None

    def interactive_coordinate_recorder(self, image_path, size=(800, 600)):
        image = Image.open(image_path)
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(image)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def on_click(self, event):
        if event.inaxes is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.coordinates.append((x, y))
            self.ax.plot(x, y, 'ro')
            self.fig.canvas.draw()
            print(f"Recorded coordinate: (x={x}, y={y})")

    def extract_features(self, image):
        return self.extractor.extract(image)

    def match_features(self, feats0, feats1):
        return self.matcher({"image0": feats0, "image1": feats1})

    def get_features_at_coordinates(self, image):
        feats = self.extract_features(image.to(self.device))
        keypoints = feats['keypoints'][0]  # Assuming batch size of 1
        descriptors = feats['descriptors'][0]
        keypoint_scores = feats['keypoint_scores'][0]

        affordance_feats = {} # following the format of the output of extract_features
        affordance_keypoints = []
        affordance_descriptors = []
        affordance_keypoint_scores = []
        
        for (x, y) in self.coordinates:
            # Find the nearest keypoint
            distances = torch.sqrt(torch.sum((keypoints - torch.tensor([x, y]).to(self.device)) ** 2, dim=1))
            min_dist_idx = torch.argmin(distances)
            affordance_keypoints.append(keypoints[min_dist_idx])
            affordance_descriptors.append(descriptors[min_dist_idx])
            affordance_keypoint_scores.append(keypoint_scores[min_dist_idx])
        
        affordance_feats['keypoints'] = torch.stack(affordance_keypoints).unsqueeze(0)
        affordance_feats['keypoint_scores'] = torch.stack(affordance_keypoint_scores).unsqueeze(0)
        affordance_feats['descriptors'] = torch.stack(affordance_descriptors).unsqueeze(0)
        affordance_feats['image_size'] = feats['image_size']

        return affordance_feats

    def identify_features_in_second_image(self, image):
        feats = self.extract_features(image.to(self.device))
        return feats

# Usage
image_matcher = ImageMatcher()
image0 = load_image('./images/cup_close1.png')
image1 = load_image('./images/cup_close2.png')

image_matcher.interactive_coordinate_recorder('./images/cup_close1.png')
feats0 = image_matcher.get_features_at_coordinates(image0)
feats1 = image_matcher.identify_features_in_second_image(image1)

matches01 = image_matcher.matcher({"image0": feats0, "image1": feats1})

feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]


axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

plt.show()

