"""
Ederest — Edge Vision for On-Prem ERP
Script 4.4 : Prédiction sur le batch suivant (Active Learning)
- Utilise le modèle YOLO fine-tuné pour auto-annoter le batch suivant
- Génère les fichiers .txt YOLO dans batch_X/labels/
- Les annotations sont ensuite corrigées sur makesense.ai
  (beaucoup moins de corrections que Grounding DINO !)

CONTEXTE ACTIVE LEARNING :
    Iteration 1 : Grounding DINO annote batch_1 → correction → YOLO v1
    Iteration 2 : YOLO v1 prédit batch_2 → correction → YOLO v2
    Iteration 3 : YOLO v2 prédit batch_3 → correction → YOLO v3
    ...
    → Chaque itération = moins de corrections humaines nécessaires

ENVIRONNEMENT : Google Colab avec GPU T4 
"""

from ultralytics import YOLO
import shutil
import os

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_VERSION  = 'v1'    # ← version du modèle à utiliser
NEXT_BATCH     = 2       # ← numéro du batch à prédire

MODEL_PATH  = f'/content/drive/MyDrive/ederest_{MODEL_VERSION}_best.pt'
INPUT_DIR   = f'./batches/batch_{NEXT_BATCH}/screens/'
OUTPUT_DIR  = f'./batches/batch_{NEXT_BATCH}/labels/'

# Seuil de confiance — plus élevé = moins de faux positifs
CONFIDENCE_THRESHOLD = 0.3

# ============================================================
# PRÉDICTION
# ============================================================
print(f"Modèle : yolo_v11_{MODEL_VERSION}_best.pt")
print(f"Batch à prédire : batch_{NEXT_BATCH}")
print(f"Output : {OUTPUT_DIR}\n")

model = YOLO(MODEL_PATH)

results = model.predict(
    source=INPUT_DIR,
    conf=CONFIDENCE_THRESHOLD,
    save_txt=True,
    save_conf=False,
    name=f'batch_{NEXT_BATCH}_predictions'
)

# ============================================================
# COPIER LES LABELS VERS LE DOSSIER DU BATCH
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRED_LABELS_DIR = f'runs/detect/batch_{NEXT_BATCH}_predictions/labels/'

if not os.path.exists(PRED_LABELS_DIR):
    print("Aucun label généré — vérifie que les images contiennent des éléments UI")
else:
    txt_files = [f for f in os.listdir(PRED_LABELS_DIR) if f.endswith('.txt')]

    for txt_file in txt_files:
        shutil.copy(
            os.path.join(PRED_LABELS_DIR, txt_file),
            os.path.join(OUTPUT_DIR, txt_file)
        )

    print(f" {len(txt_files)} fichiers de labels générés dans {OUTPUT_DIR}")

print(f"\n Prochaines étapes :")
print(f"   1. Télécharger batch_{NEXT_BATCH}/ localement")
print(f"   2. Corriger les annotations sur makesense.ai")
print(f"      (moins de corrections qu'avec Grounding DINO !)")
print(f"   3. Ajouter batch_{NEXT_BATCH} dans 1_train_val_split.py")
print(f"   4. Re-lancer 2_finetune_yolo.py avec MODEL_VERSION = 'v{int(MODEL_VERSION[1:])+1}'")