import json
import os

# 1. Chargement sécurisé
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Extraction avec FILTRAGE du None
all_frames = []
for session in data.get('practice_sessions', []):
    for collection in session.get('collections', []):
        for img_obj in collection.get('interactive_images', []):
            # On vérifie que 'image' existe et n'est pas None
            if img_obj.get('image') is not None:
                all_frames.append(img_obj)
            else:
                print(f"Image ignorée (null) dans la collection: {collection.get('title')}")

print(f"Nombre total d'images valides trouvées : {len(all_frames)}")

# 3. Split (on adapte le split au nombre réel d'images)
split_idx = int(len(all_frames) * 0.8)
train_frames = all_frames[:split_idx]
test_frames = all_frames[split_idx:]

def save_coco(frames, output_name):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "field"}]
    }
    
    for i, frame in enumerate(frames):
        img_id = i
        # Sécurité supplémentaire ici
        file_path = frame['image']
        
        coco["images"].append({
            "id": img_id,
            "file_name": os.path.basename(file_path),
            "width": 1920,
            "height": 1080
        })
        
        for field in frame.get('fields', []):
            pos = field['position']
            # COCO format: [x_min, y_min, width, height]
            bbox = [
                (pos['x'] / 100) * 1920,
                (pos['y'] / 100) * 1080,
                (pos['width'] / 100) * 1920,
                (pos['height'] / 100) * 1080
            ]
            coco["annotations"].append({
                "id": len(coco["annotations"]),
                "image_id": img_id,
                "category_id": 0,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            
    with open(output_name, 'w') as f:
        json.dump(coco, f, indent=4)
    print(f"Fichier {output_name} généré ({len(frames)} images)")

# 4. Génération
save_coco(train_frames, 'annotations_train.json')
save_coco(test_frames, 'annotations_test.json')