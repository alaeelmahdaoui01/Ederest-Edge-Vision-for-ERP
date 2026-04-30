"""
Ederest — Edge Vision for On-Prem ERP
Script 5 : Fine-tuning final YOLOv11s sur toute la data
- Fusionne tous les batches annotés et corrigés
- Lance un fine-tuning final avec plus d'epochs
- Evalue les métriques finales par classe (mAP50, précision, recall)
- Sauvegarde le modèle final sur Google Drive

CONTEXTE : Ce script est lancé une fois que tous les batches
ont été annotés et corrigés via la boucle Active Learning.
C'est le modèle final qui sera déployé en production.

ENVIRONNEMENT : Google Colab avec GPU T4 (recommandé)
INSTALLATION :
    pip install ultralytics
"""

from ultralytics import YOLO
import shutil
import os


# Tous les batches annotés et corrigés
BATCHES = [
    {
        'screens': '../data/batches/batch_1/screens/',
        'labels':  '../data/batches/batch_1/labels/'
    },
    {
        'screens': '../data/batches/batch_2/screens/',
        'labels':  '../data/batches/batch_2/labels/'
    },
    {
        'screens': '../data/batches/batch_3/screens/',
        'labels':  '../data/batches/batch_3/labels/'
    },
    {
        'screens': '../data/batches/batch_4/screens/',
        'labels':  '../data/batches/batch_4/labels/'
    },
    {
        'screens': '../data/batches/batch_5/screens/',
        'labels':  '../data/batches/batch_5/labels/'
    },
    {
        'screens': '../data/batches/batch_6/screens/',
        'labels':  '../data/batches/batch_6/labels/'
    },
    {
        'screens': '../data/batches/batch_7/screens/',
        'labels':  '../data/batches/batch_7/labels/'
    },
    {
        'screens': '../data/batches/batch_8/screens/',
        'labels':  '../data/batches/batch_8/labels/'
    },
]

DATASET_DIR = '../data/'
TRAIN_RATIO = 0.85
RANDOM_SEED = 42

# Hyperparamètres final training
EPOCHS     = 150   # plus d'epochs que les iterations intermédiaires
BATCH_SIZE = 16
IMG_SIZE   = 640
PATIENCE   = 20

# Sauvegarde
MODEL_NAME  = 'ederest_final'
DRIVE_PATH  = f'/content/drive/MyDrive/yolo_v11_best.pt'

# ============================================================
# ÉTAPE 1 — PRÉPARER LE DATASET FINAL
# ============================================================
import random

TRAIN_IMG = os.path.join(DATASET_DIR, 'images/train/')
VAL_IMG   = os.path.join(DATASET_DIR, 'images/val/')
TRAIN_LBL = os.path.join(DATASET_DIR, 'labels/train/')
VAL_LBL   = os.path.join(DATASET_DIR, 'labels/val/')

for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    os.makedirs(d, exist_ok=True)

# Créer le data.yaml
yaml_content = f"""path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val

nc: 6
names:
  0: button
  1: checkbox
  2: datepicker
  3: dropdown
  4: radio button
  5: textbox
"""

with open(os.path.join(DATASET_DIR, 'data.yaml'), 'w') as f:
    f.write(yaml_content)
print("data.yaml créé !")

# Collecter toutes les paires image+label
all_pairs = []
for batch in BATCHES:
    if not os.path.exists(batch['screens']):
        print(f"Introuvable : {batch['screens']}")
        continue
    imgs = [f for f in os.listdir(batch['screens'])
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img in imgs:
        base     = img.rsplit('.', 1)[0]
        lbl_path = os.path.join(batch['labels'], base + '.txt')
        if os.path.exists(lbl_path):
            all_pairs.append((
                os.path.join(batch['screens'], img),
                lbl_path
            ))

print(f"Total dataset final : {len(all_pairs)} images")

# Split train/val
random.seed(RANDOM_SEED)
random.shuffle(all_pairs)
split       = int(len(all_pairs) * TRAIN_RATIO)
train_pairs = all_pairs[:split]
val_pairs   = all_pairs[split:]

print(f"Train : {len(train_pairs)} images")
print(f"Val   : {len(val_pairs)} images")

# Copier les fichiers
print("\nCopie des fichiers...")
for img_path, lbl_path in train_pairs:
    shutil.copy(img_path, TRAIN_IMG)
    shutil.copy(lbl_path, TRAIN_LBL)
for img_path, lbl_path in val_pairs:
    shutil.copy(img_path, VAL_IMG)
    shutil.copy(lbl_path, VAL_LBL)

print("Dataset final prêt !")

# ============================================================
# ÉTAPE 2 — FINE-TUNING FINAL
# ============================================================
print(f"\nDémarrage fine-tuning final YOLOv11s")
print(f"   Dataset  : {len(all_pairs)} images")
print(f"   Epochs   : {EPOCHS}")
print(f"   Patience : {PATIENCE}\n")

model = YOLO('yolo11s.pt')

results = model.train(
    data=os.path.join(DATASET_DIR, 'data.yaml'),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name=MODEL_NAME,
    patience=PATIENCE,
    save=True,
    plots=True
)

print(f"\nFine-tuning final terminé !")

# ============================================================
# ÉTAPE 3 — ÉVALUATION FINALE
# ============================================================
print("\nÉvaluation du modèle final...")

best_model_path = f'runs/detect/{MODEL_NAME}/weights/best.pt'
model_eval      = YOLO(best_model_path)

metrics = model_eval.val(
    data=os.path.join(DATASET_DIR, 'data.yaml'),
    split='val'
)

# Afficher les métriques par classe
CLASS_NAMES = ['button', 'checkbox', 'datepicker', 'dropdown', 'radio button', 'textbox']

print("\n" + "="*55)
print("  RÉSULTATS FINAUX — YOLOv11s fine-tuné sur SAP GUI")
print("="*55)
print(f"  {'Classe':<15} {'mAP50':>8} {'Précision':>10} {'Recall':>8}")
print("-"*55)

for i, class_name in enumerate(CLASS_NAMES):
    try:
        map50 = metrics.box.maps[i]
        p     = metrics.box.p[i]
        r     = metrics.box.r[i]
        print(f"  {class_name:<15} {map50:>8.3f} {p:>10.3f} {r:>8.3f}")
    except:
        print(f"  {class_name:<15} {'N/A':>8} {'N/A':>10} {'N/A':>8}")

print("-"*55)
print(f"  {'GLOBAL':<15} {metrics.box.map50:>8.3f}")
print("="*55)

# ============================================================
# ÉTAPE 4 — SAUVEGARDE SUR GOOGLE DRIVE
# ============================================================
if os.path.exists('/content/drive/MyDrive/'):
    shutil.copy(best_model_path, DRIVE_PATH)
    print(f"\nModèle final sauvegardé : {DRIVE_PATH}")
else:
    print(f"\n⚠️  Drive non monté — modèle disponible : {best_model_path}")

print(f"\nProchaines étapes :")
print(f"   1. Exporter en ONNX pour déploiement")
print(f"   2. Quantization pour optimiser la taille")
print(f"   3. Tests de latence — objectif < 200ms sur CPU")