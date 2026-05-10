import mss
import numpy as np
import mss.tools
from pynput import mouse
import os
from datetime import datetime
import time
import cv2
import pygetwindow as gw
import json

# 1. Dossier de stockage
output_dir = "test_screens_ederest"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- NOUVELLE VARIABLE DE MÉMOIRE ---
# On stocke ici la dernière image pour comparer
last_screenshot_gray = None
DIFF_THRESHOLD = 1.0  # Seuil de différence en % (1.0% de changement minimum pour sauver)

print(f"🚀 Script Ederest activé (Filtre de Doublons + Focus SAP)")
print("🖱️ En attente de clics... (Ctrl+C pour arrêter)")

def get_sap_window():
    try:
        windows = gw.getWindowsWithTitle('SAP')
        if windows:
            sap_win = windows[0]
            if sap_win.visible and not sap_win.isMinimized:
                return {
                    "top": sap_win.top, "left": sap_win.left,
                    "width": sap_win.width, "height": sap_win.height,
                    "title": sap_win.title
                }
    except Exception as e:
        print(f"⚠️ Erreur fenêtre: {e}")
    return None

def is_significantly_different(new_img_gray):
    """Fonctionnalité 4: Comparaison de contenu"""
    global last_screenshot_gray

    if last_screenshot_gray is None:
        last_screenshot_gray = new_img_gray
        return True # Première capture, on accepte

    # Calcul de la différence absolue entre les deux images
    # Les images doivent avoir la même taille (géré par le cadrage SAP)
    if new_img_gray.shape != last_screenshot_gray.shape:
        last_screenshot_gray = new_img_gray
        return True

    diff = cv2.absdiff(last_screenshot_gray, new_img_gray)
    non_zero_count = np.count_nonzero(diff > 25) # On compte les pixels qui ont changé significativement
    percent_diff = (non_zero_count / diff.size) * 100

    if percent_diff > DIFF_THRESHOLD:
        print(f"📊 Changement détecté : {percent_diff:.2f}%")
        last_screenshot_gray = new_img_gray
        return True

    return False

def capture_and_save(x, y, button_type):
    with mss.mss() as sct:
        region = get_sap_window()
        if not region:
            print("❌ SAP non visible.")
            return

        screenshot = sct.grab(region)

        # Conversion rapide en gris pour la comparaison
        img_np = np.array(screenshot)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # --- LOGIQUE DE FILTRAGE ---
        if not is_significantly_different(img_gray):
            print("⏭️ Capture ignorée (Contenu identique ou trop proche).")
            return

        timestamp = datetime.now().strftime("%H-%M-%S_%f")
        base_name = f"sap_step_{timestamp}"
        img_path = os.path.join(output_dir, f"{base_name}.jpg")

        # Scaling et point rouge
        ratio_x = screenshot.width / region['width']
        ratio_y = screenshot.height / region['height']
        rel_x = int((x - region['left']) * ratio_x)
        rel_y = int((y - region['top']) * ratio_y)

        #On ne dessine le point de clic sur l'image pour la sauvegarde pour ne pas polluer yolo.
        #img_to_save = img_bgr.copy()
        #cv2.circle(img_to_save, (rel_x, rel_y), radius=15, color=(0, 0, 255), thickness=2)

        # Sauvegarde
        cv2.imwrite(img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        metadata = {
            "timestamp": timestamp,
            "click_relative": {"x": rel_x, "y": rel_y},
            "image_file": f"{base_name}.jpg"
        }

        with open(os.path.join(output_dir, f"{base_name}.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"✅ État sauvegardé : {base_name}")

def on_click(x, y, button, pressed):
    if pressed:
        #time.sleep(0.2) # On laisse l'écran se rafraîchir
        capture_and_save(x, y, button)

with mouse.Listener(on_click=on_click) as listener:
    try:
        listener.join()
    except KeyboardInterrupt:
        print("\n🛑 Script arrêté.")