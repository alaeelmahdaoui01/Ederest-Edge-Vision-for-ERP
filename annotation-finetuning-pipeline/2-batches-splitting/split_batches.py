"""
Ederest — Edge Vision for On-Prem ERP
Split du dataset en batches équilibrés (images uniquement)
- Mélange Classical Theme et Blue Crystal Theme
- Chaque batch contient le même ratio des 2 thèmes
- Les labels seront générés ensuite par Grounding DINO ou YOLO

Structure output :
    batches/
        batch_1/
            screens/   ← images uniquement, pas de labels
        batch_2/
            screens/
        ...
"""

import os
import shutil
import random


CLASSICAL_DIR    = "../data/classictheme/"
BLUE_CRYSTAL_DIR = "../data/bluecrystaltheme/"
OUTPUT_DIR       = "../data/batches/"
BATCH_SIZE       = 150
RANDOM_SEED      = 42


random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger les images des 2 thèmes
classical_imgs = [f for f in os.listdir(CLASSICAL_DIR)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
blue_imgs      = [f for f in os.listdir(BLUE_CRYSTAL_DIR)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Classical Theme    : {len(classical_imgs)} images")
print(f"Blue Crystal Theme : {len(blue_imgs)} images")

# Mélanger
random.shuffle(classical_imgs)
random.shuffle(blue_imgs)

# Calculer le ratio pour équilibrer chaque batch
total            = len(classical_imgs) + len(blue_imgs)
blue_ratio       = len(blue_imgs) / total
blue_per_batch   = int(BATCH_SIZE * blue_ratio)
classical_per_batch = BATCH_SIZE - blue_per_batch
n_batches        = total // BATCH_SIZE
remainder        = total % BATCH_SIZE

print(f"\n{n_batches} batches de {BATCH_SIZE} images")
print(f" → {classical_per_batch} Classical + {blue_per_batch} Blue Crystal par batch")
if remainder:
    print(f"   + 1 batch final de {remainder} images")
print(f"\nLes labels seront ajoutés après annotation (Grounding DINO ou YOLO)\n")

# Créer les batches
for i in range(n_batches):
    batch_screens_dir = os.path.join(OUTPUT_DIR, f"batch_{i+1}", "screens")
    os.makedirs(batch_screens_dir, exist_ok=True)

    b_imgs = blue_imgs[i*blue_per_batch:(i+1)*blue_per_batch]
    c_imgs = classical_imgs[i*classical_per_batch:(i+1)*classical_per_batch]

    for img in b_imgs:
        shutil.copy(os.path.join(BLUE_CRYSTAL_DIR, img),
                    os.path.join(batch_screens_dir, img))
    for img in c_imgs:
        shutil.copy(os.path.join(CLASSICAL_DIR, img),
                    os.path.join(batch_screens_dir, img))

    print(f"batch_{i+1} → {len(b_imgs) + len(c_imgs)} images")

# Batch final avec le reste
if remainder:
    i = n_batches
    batch_screens_dir = os.path.join(OUTPUT_DIR, f"batch_{i+1}", "screens")
    os.makedirs(batch_screens_dir, exist_ok=True)

    remaining_blue      = blue_imgs[n_batches*blue_per_batch:]
    remaining_classical = classical_imgs[n_batches*classical_per_batch:]

    for img in remaining_blue:
        shutil.copy(os.path.join(BLUE_CRYSTAL_DIR, img),
                    os.path.join(batch_screens_dir, img))
    for img in remaining_classical:
        shutil.copy(os.path.join(CLASSICAL_DIR, img),
                    os.path.join(batch_screens_dir, img))

    print(f"batch_{i+1} → {len(remaining_blue) + len(remaining_classical)} images (reste)")

print(f"\nSplit terminé !")
print(f"Prochaine étape : lancer Grounding DINO sur batch_1/screens/")