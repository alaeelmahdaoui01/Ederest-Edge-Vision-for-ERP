"""
Ederest — Edge Vision for On-Prem ERP
Fine-tuning YOLOv11s
- Fine-tune YOLOv11s sur le dataset annoté (total des batches annotés qu'on a split en train and validation sets)
- Sauvegarde le meilleur modèle
- A lancer après 1_train_val_split.py

ENVIRONNEMENT : Google Colab avec GPU T4 
INSTALLATION :
    pip install ultralytics

STRATÉGIE ACTIVE LEARNING :
    - Iteration 1 : fine-tune sur batch_1 annoté manuellement
    - Iteration 2 : fine-tune sur batch_1 + batch_2
    - Iteration N : fine-tune sur tous les batches cumulés
    → Chaque itération améliore le modèle qui annote le batch suivant
"""

from ultralytics import YOLO
import shutil
import os


DATASET_YAML  = '../data/data.yaml'
MODEL_VERSION = 'v1'        # ← incrémenter à chaque itération
EPOCHS        = 100
BATCH_SIZE    = 16
IMG_SIZE      = 640
PATIENCE      = 15          # early stopping si pas d'amélioration

# Sauvegarde du modèle sur Google Drive
SAVE_TO_DRIVE = True
DRIVE_PATH    = f'/content/drive/MyDrive/ederest_{MODEL_VERSION}_best.pt'


# FINE-TUNING
print(f"Démarrage fine-tuning YOLOv11s — version {MODEL_VERSION}")
print(f"   Dataset  : {DATASET_YAML}")
print(f"   Epochs   : {EPOCHS}")
print(f"   Patience : {PATIENCE} (early stopping)\n")

# Charger YOLOv11s pré-entraîné sur COCO
model = YOLO('yolo11s.pt')

# Lancer le fine-tuning
results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name=f'ederest_{MODEL_VERSION}',
    patience=PATIENCE,
    save=True,
    plots=True
)

print(f"\nFine-tuning terminé !")
print(f"   Meilleur modèle : runs/detect/ederest_{MODEL_VERSION}/weights/best.pt")


# SAUVEGARDE SUR GOOGLE DRIVE
best_model_path = f'runs/detect/ederest_{MODEL_VERSION}/weights/best.pt'

if SAVE_TO_DRIVE and os.path.exists('/content/drive/MyDrive/'):
    shutil.copy(best_model_path, DRIVE_PATH)
    print(f"Modèle sauvegardé sur Drive : {DRIVE_PATH}")
else:
    print(f"Drive non monté — modèle disponible localement : {best_model_path}")

print(f"\nProchaine étape : lancer 3_predict_next_batch.py")