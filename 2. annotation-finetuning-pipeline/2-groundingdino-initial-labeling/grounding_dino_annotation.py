"""
Ederest — Edge Vision for On-Prem ERP
Auto-annotation initiale avec Grounding DINO
- Prend les images d'un batch (screens/)
- Détecte automatiquement les éléments UI SAP
- Génère les fichiers .txt YOLO dans batch_X/labels/
- Les annotations sont ensuite corrigées manuellement sur makesense.ai

CONTEXTE : Grounding DINO est un modèle zero-shot — il détecte des objets
à partir de descriptions texte sans avoir été entraîné sur SAP.
Résultats imparfaits sur les interfaces SAP → correction humaine obligatoire.
Mais ça reste 5x plus rapide que d'annoter from scratch !

ENVIRONNEMENT : Google Colab avec GPU T4 
INSTALLATION :
    pip install torch torchvision transformers supervision
"""

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import os


BATCH_NUMBER = 1   # ← change ce numéro pour chaque batch
IMAGE_DIR  = f"../data/batches/batch_{BATCH_NUMBER}/screens/"
OUTPUT_DIR = f"../data/batches/batch_{BATCH_NUMBER}/labels/"

# Seuil de confiance — plus élevé = moins de faux positifs
CONFIDENCE_THRESHOLD = 0.35

# Filtres taille des bounding boxes
MAX_BOX_RATIO = 0.8   # ignorer boxes > 80% écran (faux positifs)
MIN_BOX_RATIO = 0.01  # ignorer boxes < 1% écran (bruit)

# Classes SAP — IDs YOLO correspondants
CLASSES = {
    0: "button",
    1: "checkbox",
    2: "datepicker",
    3: "dropdown",
    4: "radio button",
    5: "textbox",
}


# CHARGEMENT DU MODÈLE
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
if device == "cpu":
    print("⚠️  GPU non disponible — le script sera lent (~2-5 min/image)")

model_id  = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("Grounding DINO chargé !")


# FONCTION D'ANNOTATION
def annotate_image(image_path, output_dir):
    """
    Annote une image SAP avec Grounding DINO.

    Stratégie : une passe par classe (évite que DINO tout labellise
    comme class 0 quand on passe toutes les classes en un seul prompt).

    Returns:
        int : nombre d'éléments détectés
    """
    image = Image.open(image_path).convert("RGB")
    w, h  = image.size

    all_yolo_lines = []

    for class_id, class_name in CLASSES.items():
        prompt = class_name + "."

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[(h, w)],
            threshold=CONFIDENCE_THRESHOLD
        )[0]

        boxes  = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            # Filtrer faux positifs
            if bw > MAX_BOX_RATIO or bh > MAX_BOX_RATIO:
                print(f" {class_name} ignoré — box trop grande ({bw:.2f}x{bh:.2f})")
                continue
            if bw < MIN_BOX_RATIO or bh < MIN_BOX_RATIO:
                print(f" {class_name} ignoré — box trop petite")
                continue

            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h

            all_yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
            )
            print(f" {class_name} — score: {score:.2f}, taille: {bw:.2f}x{bh:.2f}")

    # Sauvegarder le fichier .txt YOLO
    base_name = os.path.basename(image_path).rsplit('.', 1)[0]
    out_path  = os.path.join(output_dir, base_name + '.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(all_yolo_lines))

    return len(all_yolo_lines)


# LANCER SUR LE BATCH
os.makedirs(OUTPUT_DIR, exist_ok=True)

imgs = [f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"\n{len(imgs)} images à annoter dans batch_{BATCH_NUMBER}...")
print(f"Labels sauvegardés dans : {OUTPUT_DIR}\n")


total_detections = 0
for idx, img_file in enumerate(imgs):
    print(f"\n[{idx+1}/{len(imgs)}] {img_file}")
    img_path = os.path.join(IMAGE_DIR, img_file)
    n = annotate_image(img_path, OUTPUT_DIR)
    total_detections += n
    print(f"  → {n} éléments détectés")

print(f"\nBatch {BATCH_NUMBER} annoté !")
print(f"   {len(imgs)} images | {total_detections} détections totales")
print(f"\n  Prochaine étape :")
print(f"   1. Télécharger batch_{BATCH_NUMBER}/labels/ localement")
print(f"   2. Corriger les annotations sur makesense.ai")
print(f"   3. Lancer le fine-tuning YOLO")