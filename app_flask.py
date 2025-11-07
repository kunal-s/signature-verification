import os
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pickle
from model import SiameseResNet
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/siamese_resnet.pth"
embeddings_path = "models/student_embeddings.pkl"
threshold = 0.6

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure students folder exists (it should) and create uploads dir inside static for persistence if needed
os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)

# Load model
model = SiameseResNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load embeddings
with open(embeddings_path, "rb") as f:
    student_embeddings = pickle.load(f)

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.forward_once(img)
    return emb

def calculate_match_score(distance):
    """Convert distance to a 1-10 score (inverted and scaled)"""
    # Adjust these values based on your model's typical distance range
    max_distance = 2.0
    min_distance = 0.0
    
    # Clamp distance to our expected range
    distance = max(min(distance, max_distance), min_distance)
    
    # Convert to score (inverted, so smaller distance = higher score)
    score = 10 * (1 - (distance - min_distance) / (max_distance - min_distance))
    return round(score, 1)

@app.route('/')
def index():
    # Get list of clients (previously called students)
    clients = list(student_embeddings.keys())
    return render_template('index.html', clients=clients)

@app.route('/verify', methods=['POST'])
def verify_signature():
    if 'signature' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['signature']
    client = request.form.get('client')
    
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('index'))
    
    if not client:
        flash('Please select a client', 'danger')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get embeddings and calculate distance
            query_emb = get_embedding(filepath)
            client_emb = student_embeddings[client].to(device)
            distance = F.pairwise_distance(query_emb, client_emb).item()
            
            # Convert distance to a 1-10 score
            score = calculate_match_score(distance)
            
            # Return result (keep uploaded file so we can display it)
            clients = list(student_embeddings.keys())
            result = {'score': score, 'distance': distance, 'client': client, 'uploaded_filename': filename}
            return render_template('index.html', clients=clients, result=result)
            
        except Exception as e:
            flash(f'Error processing signature: {str(e)}', 'danger')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a JPG or PNG image.', 'danger')
    return redirect(url_for('index'))


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files from the uploads folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/student_image/<client>')
def student_image(client):
    """Return a representative image for a client (first image found)."""
    client_dir = os.path.join(app.root_path, 'students', client)
    if not os.path.isdir(client_dir):
        return ("", 404)
    # find first image file
    for f in os.listdir(client_dir):
        if f.lower().endswith(('jpg', 'jpeg', 'png')):
            return send_from_directory(client_dir, f)
    return ("", 404)

if __name__ == '__main__':
    app.run(debug=True)