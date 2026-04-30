import albumentations as A
import cv2
import os

# ⚙️ CONFIG
FOLDERS = [
    "../data/classictheme/",
    "../data/bluecrystaltheme/"
]
AUGMENT_PER_IMAGE = 10

# Pipeline d'augmentation adapté SAP
augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.ImageCompression(quality_lower=75, quality_upper=100, p=0.4),
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.Sharpen(p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    # Pas de flip horizontal — SAP non symétrique
    # Pas de rotation forte — textes illisibles
])

for folder in FOLDERS:
    if not os.path.exists(folder):
        print(f"Dossier introuvable : {folder}")
        continue

    imgs = [f for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"{folder} — {len(imgs)} images trouvées")

    for img_file in imgs:
        base_name = img_file.rsplit('.', 1)[0]
        img_path = os.path.join(folder, img_file)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Impossible de lire : {img_file}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(AUGMENT_PER_IMAGE):
            augmented = augment(image=image)
            aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            new_name = f"{base_name}_aug{i}.png"
            cv2.imwrite(os.path.join(folder, new_name), aug_img)

        print(f"  ✅ {img_file} → {AUGMENT_PER_IMAGE} variantes générées")

    total = len(os.listdir(folder))
    print(f"Total dans {folder} : {total} images")

print("Augmentation terminée !")