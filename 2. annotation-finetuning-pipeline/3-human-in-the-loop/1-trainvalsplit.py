"""
Ederest — Edge Vision for On-Prem ERP
Préparation du dataset YOLO (split train/val) sur les batches where on est arrivé 
- Fusionne tous les batches annotés
- Split 85% train / 15% val
- Crée la structure attendue par YOLO :
    dataset/
        images/train/
        images/val/
        labels/train/
        labels/val/
        data.yaml
"""

import os
import shutil
import random


# Ajoute ici les batches déjà annotés et corrigés
BATCHES = [
    {
        'screens': '../data/batches/batch_1/screens/',
        'labels':  '../data/batches/batch_1/labels/'
    },
    # {
    #     'screens': '../data/batches/batch_2/screens/',
    #     'labels':  '../data/batches/batch_2/labels/'
    # },
    # ← ajoute les autres batches au fur et à mesure
]

DATASET_DIR = '../data'
TRAIN_RATIO = 0.85
RANDOM_SEED = 42


TRAIN_IMG = os.path.join(DATASET_DIR, 'images/train/')
VAL_IMG   = os.path.join(DATASET_DIR, 'images/val/')
TRAIN_LBL = os.path.join(DATASET_DIR, 'labels/train/')
VAL_LBL   = os.path.join(DATASET_DIR, 'labels/val/')

for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    os.makedirs(d, exist_ok=True)

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



# COLLECTER TOUTES LES PAIRES IMAGE+LABEL
all_pairs = []

for batch in BATCHES:
    if not os.path.exists(batch['screens']):
        print(f"Dossier introuvable : {batch['screens']}")
        continue

    imgs = [f for f in os.listdir(batch['screens'])
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img in imgs:
        base = img.rsplit('.', 1)[0]
        txt  = base + '.txt'
        lbl_path = os.path.join(batch['labels'], txt)

        if os.path.exists(lbl_path):
            all_pairs.append((
                os.path.join(batch['screens'], img),
                lbl_path
            ))
        else:
            print(f"Pas de label pour : {img} — ignoré")

print(f"\n{len(all_pairs)} paires image+label trouvées")


# SPLIT TRAIN / VAL
random.seed(RANDOM_SEED)
random.shuffle(all_pairs)

split       = int(len(all_pairs) * TRAIN_RATIO)
train_pairs = all_pairs[:split]
val_pairs   = all_pairs[split:]

print(f"Train : {len(train_pairs)} images ({TRAIN_RATIO*100:.0f}%)")
print(f"Val   : {len(val_pairs)} images ({(1-TRAIN_RATIO)*100:.0f}%)")


# COPIER LES FICHIERS
def copy_pairs(pairs, img_dir, lbl_dir):
    for img_path, lbl_path in pairs:
        shutil.copy(img_path, img_dir)
        shutil.copy(lbl_path, lbl_dir)

print("\nCopie des fichiers...")
copy_pairs(train_pairs, TRAIN_IMG, TRAIN_LBL)
copy_pairs(val_pairs,   VAL_IMG,   VAL_LBL)

print(f"\nDataset prêt dans : {DATASET_DIR}")
print(f"Prochaine étape : lancer 2_finetune_yolo.py")