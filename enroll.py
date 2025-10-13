import torch
from torchvision import transforms
from PIL import Image
import os
import pickle
from model import SiameseResNet
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Model
# ----------------------------
model_path = "models/siamese_resnet.pth"
model = SiameseResNet().to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# Transform
# ----------------------------
base_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

augment_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(128, scale=(0.9,1.0)),
    transforms.RandomHorizontalFlip(),
])

# ----------------------------
# Enroll Students
# ----------------------------
students_dir = "students"
student_embeddings = {}
num_augmentations = 5  # number of augmented samples per original

with torch.no_grad():
    for student in os.listdir(students_dir):
        student_path = os.path.join(students_dir, student)
        if not os.path.isdir(student_path):
            continue

        embeddings = []
        for f in os.listdir(student_path):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(student_path, f)
                img = Image.open(img_path).convert("RGB")
                img_tensor = base_transform(img).unsqueeze(0).to(device)
                emb = model.forward_once(img_tensor)
                embeddings.append(emb.cpu())

                # Augmentations
                for _ in range(num_augmentations):
                    img_aug = augment_transform(img)
                    img_aug = base_transform(img_aug).unsqueeze(0).to(device)
                    emb_aug = model.forward_once(img_aug)
                    embeddings.append(emb_aug.cpu())

        if embeddings:
            mean_emb = torch.stack(embeddings).mean(dim=0)
            student_embeddings[student] = mean_emb

# ----------------------------
# Save embeddings
# ----------------------------
os.makedirs("models", exist_ok=True)
with open("models/student_embeddings.pkl", "wb") as f:
    pickle.dump(student_embeddings, f)

print(f"✅ Enrolled {len(student_embeddings)} students with augmentation")
