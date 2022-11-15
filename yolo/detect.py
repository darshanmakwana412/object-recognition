import torch
from models import load_model
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
from PIL import Image

from utils.utils import non_max_suppression

# def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
#     """Draws detections in output image and stores this.
#     :param image_path: Path to input image
#     :type image_path: str
#     :param detections: List of detections on image
#     :type detections: [Tensor]
#     :param img_size: Size of each image dimension for yolo
#     :type img_size: int
#     :param output_path: Path of output directory
#     :type output_path: str
#     :param classes: List of class names
#     :type classes: [str]
#     """
#     # Create plot
#     img = np.array(Image.open(image_path))
#     plt.figure()
#     fig, ax = plt.subplots(1)
#     ax.imshow(img)
#     # Rescale boxes to original image
#     detections = rescale_boxes(detections, img_size, img.shape[:2])
#     unique_labels = detections[:, -1].cpu().unique()
#     n_cls_preds = len(unique_labels)
#     # Bounding-box colors
#     cmap = plt.get_cmap("tab20b")
#     colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
#     bbox_colors = random.sample(colors, n_cls_preds)
#     for x1, y1, x2, y2, conf, cls_pred in detections:

#         print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

#         box_w = x2 - x1
#         box_h = y2 - y1

#         color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
#         # Create a Rectangle patch
#         bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
#         # Add the bbox to the plot
#         ax.add_patch(bbox)
#         # Add label
#         plt.text(
#             x1,
#             y1,
#             s=classes[int(cls_pred)],
#             color="white",
#             verticalalignment="top",
#             bbox={"color": color, "pad": 0})

#     # Save generated image with detections
#     plt.axis("off")
#     plt.gca().xaxis.set_major_locator(NullLocator())
#     plt.gca().yaxis.set_major_locator(NullLocator())
#     filename = os.path.basename(image_path).split(".")[0]
#     output_path = os.path.join(output_path, f"{filename}.png")
#     plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
#     plt.close()

class DetectDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = os.listdir(self.img_dir)

    def __getitem__(self, index):

        img_path = os.path.join(self.img_dir, self.images[index])
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)

        return img

    def __len__(self):
        return len(self.images)

def detect():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("../cfg/yolov3.cfg", "./weights/yolov3.weights")

    detectDataset = DetectDataset("../data/test")
    dataloader = DataLoader(
        detectDataset,
        batch_size=1,
    )
    
    model.eval()

    for imgs in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        imgs = imgs.permute(0, 3, 2, 1)
        with torch.no_grad():
            outputs = model(imgs)
            detections = non_max_suppression(outputs)
            print(detections)

if __name__ == "__main__":
    detect()