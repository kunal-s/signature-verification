import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from model import SiameseResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 1e-4
epochs = 30
margin = 1.0

# ----------------------------
# Dataset (pairs)
# ----------------------------
class StudentPairsDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.data = []

        students = os.listdir(root_dir)
        for student in students:
            student_path = os.path.join(root_dir, student)
            imgs = [os.path.join(student_path, f) for f in os.listdir(student_path) 
                    if f.lower().endswith((".jpg",".png",".jpeg"))]
            for i in range(len(imgs)):
                for j in range(i+1, len(imgs)):
                    self.data.append((imgs[i], imgs[j], 1))  # genuine pair

            # negative pairs
            other_students = [s for s in students if s != student]
            for _ in range(len(imgs)):
                neg_student = random.choice(other_students)
                neg_img = random.choice(os.listdir(os.path.join(root_dir, neg_student)))
                neg_img_path = os.path.join(root_dir, neg_student, neg_img)
                self.data.append((random.choice(imgs), neg_img_path, 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.data[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        if self.augment:
            aug = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(128, scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip()
            ])
            img1 = aug(img1)
            img2 = aug(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# ----------------------------
# Transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = StudentPairsDataset("students", transform=transform, augment=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------------------
# Model, Loss, Optimizer
# ----------------------------
model = SiameseResNet(pretrained=True).to(device)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        loss = torch.mean(label * dist**2 + (1-label) * torch.clamp(self.margin - dist, min=0)**2)
        return loss

criterion = ContrastiveLoss(margin)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for img1, img2, label in dataloader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

# ----------------------------
# Save model
# ----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/siamese_resnet.pth")
print("✅ Model saved to models/siamese_resnet.pth")
