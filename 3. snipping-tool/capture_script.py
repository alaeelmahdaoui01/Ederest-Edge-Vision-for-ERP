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

# --- VARIABLES DE SESSION ---
last_screenshot_gray = None
DIFF_THRESHOLD = 1.0
session_steps = []  # Liste unique qui contiendra toutes les étapes
step_counter = 1    # Compteur pour le champ "step"

print(f"🚀 Script Ederest activé (Journal unique JSON + Filtre Doublons)")
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
    global last_screenshot_gray
    if last_screenshot_gray is None:
        last_screenshot_gray = new_img_gray
        return True
    if new_img_gray.shape != last_screenshot_gray.shape:
        last_screenshot_gray = new_img_gray
        return True

    diff = cv2.absdiff(last_screenshot_gray, new_img_gray)
    non_zero_count = np.count_nonzero(diff > 25)
    percent_diff = (non_zero_count / diff.size) * 100

    if percent_diff > DIFF_THRESHOLD:
        print(f"📊 Changement détecté : {percent_diff:.2f}%")
        last_screenshot_gray = new_img_gray
        return True
    return False

def capture_and_save(x, y, button_type):
    global step_counter, session_steps

    with mss.mss() as sct:
        region = get_sap_window()
        if not region:
            print("❌ SAP non visible.")
            return

        screenshot = sct.grab(region)
        img_np = np.array(screenshot)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if not is_significantly_different(img_gray):
            print("⏭️ Capture ignorée (Doublon).")
            return

        # Nommage conforme à la structure demandée
        timestamp_str = f"step_{step_counter:03d}"
        img_name = f"sap_screen_{timestamp_str}.jpg"
        img_path = os.path.join(output_dir, img_name)

        # Calcul des coordonnées relatives
        ratio_x = screenshot.width / region['width']
        ratio_y = screenshot.height / region['height']
        rel_x = int((x - region['left']) * ratio_x)
        rel_y = int((y - region['top']) * ratio_y)

        # Sauvegarde de l'image (sans point rouge pour YOLO)
        cv2.imwrite(img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # --- AJOUT À LA STRUCTURE UNIQUE ---
        new_step = {
            "step": step_counter,
            "type": "click",
            "x": rel_x,
            "y": rel_y,
            "value": None,
            "screenshot": img_name,
            "timestamp": timestamp_str
        }

        session_steps.append(new_step)

        # Sauvegarde du fichier JSON unique à chaque étape (plus sûr)
        json_path = os.path.join(output_dir, "session_steps.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(session_steps, f, indent=4)

        print(f"✅ Étape {step_counter} enregistrée dans session_steps.json")
        step_counter += 1

def on_click(x, y, button, pressed):
    if pressed:
        # Petit délai optionnel si l'interface SAP est lente à réagir
        capture_and_save(x, y, button)

# Listener
with mouse.Listener(on_click=on_click) as listener:
    try:
        listener.join()
    except KeyboardInterrupt:
        print(f"\n🛑 Script arrêté. {len(session_steps)} étapes sauvegardées dans session_steps.json")