import mss
import numpy as np
import mss.tools
from pynput import mouse, keyboard
import os
from datetime import datetime
import time
import cv2
import pygetwindow as gw
import json

# 1. Configuration des dossiers
output_dir = "test_screens_ederest"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- VARIABLES DE SESSION ---
last_screenshot_gray = None
DIFF_THRESHOLD = 1.0
session_steps = []  
step_counter = 1    
last_action_saved = False # Flag pour savoir si le dernier clic a généré un fichier

print(f"🚀 Script Ederest Pro (Capture SAP + Annulation Echap)")
print("🖱️ Cliquez pour capturer | ⌨️ Echap pour annuler la dernière capture | Ctrl+C pour quitter")

def get_sap_window():
    """Détecte et renforce le focus sur la fenêtre SAP."""
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
    """Filtre les captures identiques (ex: clic dans le vide)."""
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
        last_screenshot_gray = new_img_gray
        return True
    return False

def capture_and_save(x, y):
    global step_counter, session_steps, last_action_saved

    with mss.mss() as sct:
        region = get_sap_window()
        if not region:
            print("❌ SAP non visible.")
            last_action_saved = False
            return

        screenshot = sct.grab(region)
        img_np = np.array(screenshot)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Vérification du changement d'écran
        if not is_significantly_different(img_gray):
            print("⏭️ Doublon détecté : aucune capture créée.")
            last_action_saved = False
            return

        # Naming & Path
        timestamp_str = f"step_{step_counter:03d}"
        img_name = f"sap_screen_{timestamp_str}.jpg"
        img_path = os.path.join(output_dir, img_name)

        # Coordonnées relatives
        ratio_x = screenshot.width / region['width']
        ratio_y = screenshot.height / region['height']
        rel_x = int((x - region['left']) * ratio_x)
        rel_y = int((y - region['top']) * ratio_y)

        # Sauvegarde image
        cv2.imwrite(img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # Mise à jour JSON
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

        json_path = os.path.join(output_dir, "session_steps.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(session_steps, f, indent=4)

        print(f"✅ Étape {step_counter} enregistrée.")
        step_counter += 1
        last_action_saved = True # On marque l'action comme annulable

def on_click(x, y, button, pressed):
    if pressed:
        capture_and_save(x, y)

def on_press(key):
    global step_counter, session_steps, last_action_saved
    
    # Gestion de l'annulation via Echap
    if key == keyboard.Key.esc:
        if last_action_saved and len(session_steps) > 0:
            # Retrait de la dernière entrée
            last_entry = session_steps.pop()
            
            # Suppression du fichier image
            img_to_delete = os.path.join(output_dir, last_entry["screenshot"])
            if os.path.exists(img_to_delete):
                os.remove(img_to_delete)
            
            # Mise à jour du fichier JSON
            json_path = os.path.join(output_dir, "session_steps.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(session_steps, f, indent=4)
            
            step_counter -= 1
            last_action_saved = False # Empêche une double annulation sur le même clic
            print(f"↩️ Annulation réussie. Retour à l'étape {step_counter}.")
        else:
            print("ℹ️ Rien à annuler pour le dernier mouvement (aucun fichier n'avait été créé).")

# Lancement des listeners
mouse_listener = mouse.Listener(on_click=on_click)
key_listener = keyboard.Listener(on_press=on_press)

mouse_listener.start()
key_listener.start()

try:
    mouse_listener.join()
    key_listener.join()
except KeyboardInterrupt:
    print(f"\n🛑 Session terminée. {len(session_steps)} étapes au total.")