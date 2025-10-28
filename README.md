# 🖋️ Signature Identification System (Siamese ResNet-based)

An intelligent **signature verification and identification system** built using a **Siamese ResNet architecture**.  
It verifies whether a given signature belongs to a registered user by comparing deep feature embeddings of signature images.

This system is optimized for **limited user data** (as few as 5 samples per user), using **data augmentation** and **contrastive learning** for robust verification.

---

## 🚀 Features

✅ **ResNet-18 based Siamese Network** – extracts deep signature embeddings.  
✅ **Contrastive Loss Training** – learns to distinguish between genuine and forged pairs.  
✅ **Automatic Embedding Enrollment** – generates and saves student signature representations.  
✅ **Real-time Verification** – compare uploaded signatures with enrolled ones.  
✅ **Augmentation Pipeline** – increases dataset size for limited samples.  
✅ **Support for CEDAR + Student Dataset** – seamlessly integrates both datasets.

---

## 🧩 Project Structure

```

Signature-Identification/
│
├── dataset.py                 # Loads and augments signature data
├── model.py                   # Defines Siamese ResNet model
├── train.py                   # Trains the model using CEDAR + student data
├── predict.py                 # Verifies a new signature against enrolled users
├── enroll.py                  # Scans test images and saves embeddings
│
├── signatures/                # Public signature dataset (already large)
│── students/                  # Contains student signatures (only 5 per user)
│       ├── student1/
│       ├── student2/
│       └── ...
│
├── models/
│   ├── student_embeddings.pkl # Saved student embeddings after enrollment
    └── siamese_resnet.pth   # Trained model weights

````

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/Signature-Identification.git
cd Signature-Identification
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---


## 🏋️‍♂️ Training the Model

Train the Siamese ResNet using both **CEDAR** and **student** signatures:

```bash
python train.py
```

This script:

* Loads and augments the datasets
* Creates genuine/forged signature pairs
* Trains the Siamese ResNet model with **contrastive loss**
* Saves model weights to `models/siamese_resnet.pth`

---

## 🧬 Enrollment (Generate Student Embeddings)

Once training is done, generate embeddings for all student signatures:

```bash
python enroll.py
```

This step:

* Loads the trained model
* Computes feature embeddings for each student's signatures
* Saves them in `mdoels/student_embeddings.pkl`

---

## 🔍 Signature Verification

Verify a given signature against all enrolled users, by configuring query_path:

```bash
python predict.py
```

The system compares the query signature with all stored embeddings and prints the most probable match, along with the similarity score.

**Example Output:**

```
Predicted Student: student5
Similarity Score: 0.912
Result: Genuine
```

---

## 🧪 Model Architecture Overview

The Siamese ResNet model uses two identical branches of ResNet-18 sharing weights.

```
        Input 1                  Input 2
           │                       │
           ▼                       ▼
      ResNet Encoder          ResNet Encoder
           │                        │
           └── Feature Embeddings ──┘
                     │
             Euclidean Distance
                     │
                Contrastive Loss
```

This allows the network to learn meaningful signature representations and distinguish genuine vs forged pairs effectively.

---

## 📈 Performance

| Dataset                | Accuracy | Loss     | Remarks                    |
| ---------------------- | -------- | -------- | -------------------------- |
| CEDAR                  | ~97%     | Low      | High-quality dataset       |
| Students (5→Augmented) | ~93%     | Moderate | Improved with augmentation |

---

## 🛠️ Technologies Used

* **Python 3.11**
* **PyTorch** – deep learning framework
* **OpenCV / PIL** – image handling
* **Albumentations** – data augmentation
* **NumPy / Pandas** – preprocessing utilities

---

## 🧾 Future Improvements

* ✅ Web Interface for Upload & Verification
* ✅ Add One-shot Learning for new users
* 🔲 Integrate Live Signature Capture

---

```
