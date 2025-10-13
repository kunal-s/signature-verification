import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import pickle
from model import SiameseResNet

# -------------------------------------------------------
# 🎨 App Configuration
# -------------------------------------------------------
st.set_page_config(page_title="Signature Verification System", page_icon="✍️", layout="centered")

st.markdown("""
    <style>
        .main-title {
            font-size: 2.2em;
            font-weight: bold;
            text-align: center;
            color: #2E86C1;
        }
        .result-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #E0E0E0;
            text-align: center;
            margin-top: 20px;
        }
        .predicted-name {
            font-size: 1.8em;
            font-weight: bold;
            color: #27AE60;
        }
        .unknown-name {
            font-size: 1.8em;
            font-weight: bold;
            color: #E74C3C;
        }
        .subtext {
            color: #7B7D7D;
            font-size: 0.9em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">✍️ Signature Verification System</p>', unsafe_allow_html=True)
st.markdown("#### Identify, Enroll, and Manage Student Signatures with Siamese Network (ResNet)")

# -------------------------------------------------------
# ⚙️ Configuration
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/siamese_resnet.pth"
EMB_PATH = "models/student_embeddings.pkl"
STUDENT_DIR = "students"

# -------------------------------------------------------
# 🧠 Helper Functions
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model = SiameseResNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

@st.cache_data
def load_embeddings():
    if not os.path.exists(EMB_PATH):
        return {}
    with open(EMB_PATH, "rb") as f:
        return pickle.load(f)

def get_transform():
    return transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def get_embedding(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = get_transform()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.forward_once(img)
    return emb

# -------------------------------------------------------
# 🧭 Sidebar Navigation
# -------------------------------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🧍 Enroll Students", "🔍 Identify Signature", "⚙️ Model Info"]
)

# -------------------------------------------------------
# 🏠 Home
# -------------------------------------------------------
if menu == "🏠 Home":
    st.header("Welcome 👋")
    st.markdown("""
    This system uses a **Siamese Neural Network (ResNet)** to verify or identify student signatures.  
    You can:
    - 🧍 **Enroll** new students and store their embeddings  
    - 🔍 **Identify** whose signature has been uploaded  
    - ⚙️ **Check** model and system info  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/685/685686.png", width=150)
    if torch.cuda.is_available():
        st.success("✅ GPU available — Running on CUDA")
    else:
        st.warning("⚙️ Running on CPU — may be slower")

# -------------------------------------------------------
# 🧍 Enroll Students
# -------------------------------------------------------
elif menu == "🧍 Enroll Students":
    st.header("🧍 Enroll a New Student")
    st.markdown("Upload one or more signature images to register a student into the system.")

    student_name = st.text_input("Enter Student Name")
    uploaded_files = st.file_uploader("Upload Signature Images", accept_multiple_files=True, type=["png","jpg","jpeg"])

    if st.button("Enroll Student"):
        if not student_name or not uploaded_files:
            st.error("⚠️ Please provide a name and at least one image.")
        else:
            os.makedirs(os.path.join(STUDENT_DIR, student_name), exist_ok=True)
            for file in uploaded_files:
                img = Image.open(file)
                save_path = os.path.join(STUDENT_DIR, student_name, file.name)
                img.save(save_path)

            model = load_model()
            student_embeddings = load_embeddings()

            embeddings = []
            for file in uploaded_files:
                img = Image.open(file).convert("RGB")
                img_tensor = get_transform()(img).unsqueeze(0).to(device)
                emb = model.forward_once(img_tensor)
                embeddings.append(emb.cpu())

            mean_emb = torch.stack(embeddings).mean(dim=0)
            student_embeddings[student_name] = mean_emb

            os.makedirs("models", exist_ok=True)
            with open(EMB_PATH, "wb") as f:
                pickle.dump(student_embeddings, f)

            st.success(f"✅ Student **{student_name}** enrolled successfully!")

# -------------------------------------------------------
# 🔍 Identify Signature
# -------------------------------------------------------
elif menu == "🔍 Identify Signature":
    st.header("🔍 Identify a Signature")
    st.markdown("Upload a signature image and the system will identify the student.")

    uploaded_query = st.file_uploader("Upload Signature Image", type=["png","jpg","jpeg"])
    threshold = st.slider("Identification Threshold", 0.1, 2.0, 0.6, 0.05)

    if uploaded_query is not None:
        st.image(uploaded_query, caption="Uploaded Signature", width=300)
        if st.button("Identify"):
            model = load_model()
            student_embeddings = load_embeddings()

            if not student_embeddings:
                st.error("⚠️ No enrolled students found! Please enroll first.")
            else:
                temp_path = "temp_query.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_query.read())
                query_emb = get_embedding(model, temp_path)

                distances = {name: F.pairwise_distance(query_emb, emb.to(device)).item()
                             for name, emb in student_embeddings.items()}

                pred_name = min(distances, key=distances.get)
                pred_dist = distances[pred_name]

                # Elegant result display
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                if pred_dist > threshold:
                    st.markdown('<p class="unknown-name">❌ Unknown Signature</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="subtext">(Distance = {pred_dist:.3f})</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="predicted-name">✅ Identified: {pred_name}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="subtext">(Distance = {pred_dist:.3f})</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("### 📊 Distance Summary")
                st.dataframe(
                    {"Student": list(distances.keys()),
                     "Distance": [round(v, 4) for v in distances.values()]}
                )

# -------------------------------------------------------
# ⚙️ Model Info
# -------------------------------------------------------
elif menu == "⚙️ Model Info":
    st.header("⚙️ Model Information")
    st.markdown(f"""
    - **Model Path:** `{MODEL_PATH}`
    - **Embeddings Path:** `{EMB_PATH}`
    - **Students Directory:** `{STUDENT_DIR}`
    - **Device:** `{device}`
    """)
    if os.path.exists(MODEL_PATH):
        st.success("✅ Model file found")
    else:
        st.error("❌ Model file not found — Train or upload it first.")
    if os.path.exists(EMB_PATH):
        st.success("✅ Embeddings file found")
    else:
        st.warning("⚠️ No embeddings found — enroll students first.")
