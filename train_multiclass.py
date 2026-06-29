import argparse
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import json


class MultiClassDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        label = int(row['label_idx'])
        return img, label


def build_model(num_classes):
    m = models.mobilenet_v2(pretrained=True)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
    return m


def train(args):
    df = pd.read_csv(args.csv)
    labels = sorted(df['label'].unique())
    label2idx = {l:i for i,l in enumerate(labels)}
    df['label_idx'] = df['label'].map(label2idx)
    df.to_csv('prepared_multiclass.csv', index=False)

    tf = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = MultiClassDataset('prepared_multiclass.csv', transforms=tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for imgs, labs in loader:
            imgs = imgs.to(device)
            labs = labs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        print(f'Epoch {epoch+1}/{args.epochs} loss:', running/len(dataset))

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir,'multiclass_model.pth'))
    with open(os.path.join(args.out_dir,'label_map.json'),'w',encoding='utf-8') as f:
        json.dump(label2idx,f,ensure_ascii=False)
    print('Saved model and label map to', args.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out-dir', default='.')
    args = parser.parse_args()
    train(args)
