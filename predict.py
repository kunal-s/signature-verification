import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pickle
from model import SiameseResNet

# ----------------------------
# Configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/siamese_resnet.pth"
embeddings_path = "models/student_embeddings.pkl"
query_path = "test_images/taslim2.jpg"  # Change to your test file
threshold = 0.6  # Adjust based on validation

# ----------------------------
# Load Model
# ----------------------------
model = SiameseResNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# Load Student Embeddings
# ----------------------------
if not os.path.exists(embeddings_path):
    raise FileNotFoundError("❌ Embedding file not found. Run enroll.py first!")

with open(embeddings_path, "rb") as f:
    student_embeddings = pickle.load(f)

if not student_embeddings:
    raise ValueError("⚠️ No student embeddings found! Re-run enroll.py with valid data.")

# ----------------------------
# Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----------------------------
# Get embedding for query
# ----------------------------
def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.forward_once(img)
    return emb

# ----------------------------
# Predict
# ----------------------------
print("🔍 Comparing query signature with enrolled students...\n")

query_emb = get_embedding(query_path)
distances = {}

for name, emb in student_embeddings.items():
    emb = emb.to(device)
    dist = F.pairwise_distance(query_emb, emb).item()
    distances[name] = dist

# Closest match
pred_name = min(distances, key=distances.get)
pred_dist = distances[pred_name]

if pred_dist > threshold:
    print(f"⚠️ Unknown Signature — No match found (distance = {pred_dist:.3f})")
else:
    print(f"✅ Predicted: {pred_name}, Distance: {pred_dist:.3f}")

# Distance summary
print("\n📊 Distance Summary:")
for name, dist in sorted(distances.items(), key=lambda x: x[1]):
    print(f"{name}: {dist:.4f}")
