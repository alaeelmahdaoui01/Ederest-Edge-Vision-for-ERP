"""
Ederest — Edge Vision for On-Prem ERP
Script 2 : Split du dataset en batches de taille fixe
- Prend toutes les images d'un seul dossier
- Les divise en batches de 150 images mélangés aléatoirement
- Chaque batch sera annoté progressivement via Human-in-the-Loop

Structure output :
    batches/
        batch_1/
            screens/   ← 150 images
        batch_2/
            screens/   ← 150 images
        ...
"""

import os
import shutil
import random

# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_DIR   = "../data/images/"        # dossier unique contenant toutes les images
OUTPUT_DIR  = "./batches/"
BATCH_SIZE  = 150
RANDOM_SEED = 42

# ============================================================
# MAIN
# ============================================================
random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger toutes les images
imgs = [f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Total images : {len(imgs)}")

# Mélanger aléatoirement
random.shuffle(imgs)

# Calculer le nombre de batches
n_batches = len(imgs) // BATCH_SIZE
remainder = len(imgs) % BATCH_SIZE

print(f"{n_batches} batches de {BATCH_SIZE} images")
if remainder:
    print(f"   + 1 batch final de {remainder} images")

# Créer les batches
for i in range(n_batches):
    batch_dir = os.path.join(OUTPUT_DIR, f"batch_{i+1}", "screens")
    os.makedirs(batch_dir, exist_ok=True)

    batch_imgs = imgs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    for img in batch_imgs:
        shutil.copy(
            os.path.join(IMAGE_DIR, img),
            os.path.join(batch_dir, img)
        )
    print(f"batch_{i+1} → {len(batch_imgs)} images")

# Batch final avec le reste
if remainder:
    i = n_batches
    batch_dir = os.path.join(OUTPUT_DIR, f"batch_{i+1}", "screens")
    os.makedirs(batch_dir, exist_ok=True)

    batch_imgs = imgs[n_batches*BATCH_SIZE:]
    for img in batch_imgs:
        shutil.copy(
            os.path.join(IMAGE_DIR, img),
            os.path.join(batch_dir, img)
        )
    print(f"batch_{i+1} → {len(batch_imgs)} images (reste)")

print(f"\nSplit terminé !")
print(f"Prochaine étape : annoter batch_1 via Grounding DINO ou manuellement")