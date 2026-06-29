import argparse
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import pandas as pd


class MultiLabelImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        labels = torch.tensor(row[['prediabetes','insulin_resistance','hypertension']].values.astype(float), dtype=torch.float32)
        return img, labels


def build_model(num_outputs=3):
    m = models.mobilenet_v2(pretrained=True)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_outputs))
    return m


def train(args):
    tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = MultiLabelImageDataset(args.csv, args.img_dir, transforms=tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(num_outputs=3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {epoch_loss:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    print('Model saved to', os.path.join(args.out_dir, 'model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--out-dir', default='.')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
