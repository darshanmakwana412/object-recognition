import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models import load_model
from utils.loss import compute_loss

import os
import numpy as np
import pandas as pd
from PIL import Image

class ListDataset(Dataset):
    def __init__(self, img_path, label_path):

        self.img_path = img_path

        boxes_data = pd.read_csv(label_path)
        self.images = boxes_data["image"].to_list()
        self.boxes = []

        for img_id, row in boxes_data.iterrows():
            Cx = (row["xmax"]+row["xmin"])/2.0
            Cy = (row["ymax"]+row["ymin"])/2.0
            width = (row["xmax"]-row["xmin"])
            height = (row["ymax"]-row["ymin"])
            box = [img_id, 1, Cx, Cy, height, width]
            self.boxes.append(box)

        self.boxes = np.array(self.boxes)

    def __getitem__(self, index):

        img_path = os.path.join(self.img_path, self.images[index])
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        box = self.boxes[index]

        return img, box

    def __len__(self):
        return len(self.images)

def _create_data_loader(img_path, label_path, batch_size):

    dataset = ListDataset(
        img_path=img_path,
        label_path=label_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100

    model = load_model("../cfg/yolov3.cfg")
    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    dataloader = _create_data_loader(
        "../data/train/",
        "../data/train_solution_bounding_boxes.csv",
        mini_batch_size,
    )

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(
        params,
        lr=model.hyperparams['learning_rate'],
        weight_decay=model.hyperparams['decay'],
    )

    for epoch in range(1, epochs+1):
        print("\n---- Training the yolo Model ----")
        model.train()

        for batch_i, (imgs, boxes) in enumerate(dataloader):

            batches_done = len(dataloader) * epoch + batch_i
            imgs = imgs.to(device, non_blocking=True)
            boxes = boxes.to(device)

            imgs = imgs.permute(0, 3, 2, 1)
            outputs = model(imgs)

            loss, loss_components = compute_loss(outputs, boxes, model)
            loss.backward()

            if batches_done % model.hyperparams['subdivisions'] == 0:

                lr = model.hyperparams['learning_rate']
                if batches_done < int(model.hyperparams['burn_in']):
                    lr *= (batches_done / int(model.hyperparams['burn_in']))
                else:
                    for threshold, value in int(model.hyperparams['lr_steps']):
                        if batches_done > threshold:
                            lr *= value

                for g in optimizer.param_groups:
                    g['lr'] = lr

                optimizer.step()
                optimizer.zero_grad()

            print(loss, loss_components)
            # checkpoint_path = f"model/yolo.pth"
            # print(f"---- Saving epoch {epoch} to: '{checkpoint_path}' ----")
            # torch.save(model.state_dict(), checkpoint_path)
        
if __name__ == "__main__":
    train()