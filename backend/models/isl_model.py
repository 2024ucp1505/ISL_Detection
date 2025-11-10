import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import time
from collections import defaultdict
import threading

# =================== SPEED PRESETS ===================
SPEED_PRESETS = {
    "beginner": {"STABLE_MS": 1200, "DEBOUNCE_MS": int(1200 * 0.30)},  # 360 ms
    "medium":   {"STABLE_MS": 1000, "DEBOUNCE_MS": int(1000 * 0.30)},  # 300 ms
    "fast":     {"STABLE_MS": 800,  "DEBOUNCE_MS": int(800  * 0.30)},  # 240 ms
}
CURRENT_SPEED = "medium"  # change live with 1/2/3
STABLE_MS   = SPEED_PRESETS[CURRENT_SPEED]["STABLE_MS"]
DEBOUNCE_MS = SPEED_PRESETS[CURRENT_SPEED]["DEBOUNCE_MS"]

SHOW_MIRROR = True  # mirror preview only

# =================== MODEL & MP ===================
model = keras.models.load_model("model.h5")
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
mp_hands   = mp.solutions.hands

alphabet = list("123456789") + list(string.ascii_uppercase)

# ---- infer model input width (supports 42/63/84/126) ----
EXPECTED_N = int(getattr(model, "input_shape", [None, 42])[-1] or 42)
NEED_XYZ   = EXPECTED_N in (63, 126)
TWO_HAND   = EXPECTED_N in (84, 126)

# =================== HELPERS ===================
def apply_speed(preset_name: str):
    global CURRENT_SPEED, STABLE_MS, DEBOUNCE_MS
    if preset_name not in SPEED_PRESETS:
        return
    CURRENT_SPEED = preset_name
    STABLE_MS   = SPEED_PRESETS[preset_name]["STABLE_MS"]
    DEBOUNCE_MS = SPEED_PRESETS[preset_name]["DEBOUNCE_MS"]
    print(f"[SPEED] {CURRENT_SPEED.upper()} → STABLE={STABLE_MS}ms  DEBOUNCE={DEBOUNCE_MS}ms")

def calc_xy(img, lms):
    h, w = img.shape[0], img.shape[1]
    out = []
    for p in lms.landmark:
        x = min(int(p.x * w), w - 1)
        y = min(int(p.y * h), h - 1)
        out.append([x, y])
    return out

def calc_xyz(img, lms):
    h, w = img.shape[0], img.shape[1]
    out = []
    for p in lms.landmark:
        x = min(int(p.x * w), w - 1)
        y = min(int(p.y * h), h - 1)
        z = float(p.z)
        out.append([x, y, z])
    return out

def pre_xy(landmark_list):
    tmp = copy.deepcopy(landmark_list)
    bx, by = tmp[0][0], tmp[0][1]
    for i in range(len(tmp)):
        tmp[i][0] -= bx
        tmp[i][1] -= by
    flat = list(itertools.chain.from_iterable(tmp))
    mx = max(map(abs, flat)) or 1.0
    return [v / mx for v in flat]

def pre_xyz(landmark_list_xyz):
    tmp = copy.deepcopy(landmark_list_xyz)
    bx, by, bz = tmp[0][0], tmp[0][1], tmp[0][2]
    for i in range(len(tmp)):
        tmp[i][0] -= bx
        tmp[i][1] -= by
        tmp[i][2] -= bz
    flat = list(itertools.chain.from_iterable(tmp))
    mx = max(map(abs, flat)) or 1.0
    return [v / mx for v in flat]

def hand_feature(img, lms, use_xyz):
    if lms is None:
        return None
    return pre_xyz(calc_xyz(img, lms)) if use_xyz else pre_xy(calc_xy(img, lms))

def build_two_hand_feature(img, left_lms, right_lms, use_xyz):
    # Per-hand feature length
    per_len = 63 if use_xyz else 42
    fL = hand_feature(img, left_lms,  use_xyz)
    fR = hand_feature(img, right_lms, use_xyz)
    if fL is None: fL = [0.0]*per_len
    if fR is None: fR = [0.0]*per_len
    # Fixed order: Left then Right
    return fL + fR

# ======== STUB: single-call autocorrect+translate (replace later) ========
def autocorrect_and_translate_single_call(word, target_lang="Hindi"):
    corrected = word
    translated = word
    return corrected, translated

def async_call(fn, *args, **kwargs):
    t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    t.start()

# =================== STATE ===================
from collections import defaultdict
chr_count = defaultdict(int)
current_max = None
stable_start = None
last_commit_time = 0.0

current_word = ""                 # raw characters (committed via stability)
corrected_sentence = ""           # appended on SPACE
translated_sentence = ""          # appended on SPACE

# =================== CAMERA ===================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera not available")

# choose speed once
try:
    sel = input("Speed? (1=beginner  2=medium  3=fast) [default 2]: ").strip()
    if sel == "1": apply_speed("beginner")
    elif sel == "3": apply_speed("fast")
    else: apply_speed("medium")
except Exception:
    apply_speed("medium")

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue
        if SHOW_MIRROR:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)

        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        vis.flags.writeable = True
        h, w = vis.shape[:2]
        now = time.monotonic()

        # -------- collect mediapipe hands --------
        left_lms, right_lms = None, None
        if results.multi_hand_landmarks:
            for hand_lms, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(
                    vis, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                label = handness.classification[0].label  # "Left"/"Right"
                if label == "Left":
                    left_lms = hand_lms
                elif label == "Right":
                    right_lms = hand_lms

        # -------- unified prediction (single- or two-hand model) --------
        pred_label, pred_conf = "-", 0.0

        if TWO_HAND:
            # Model expects concatenated (L,R) features; pad missing hand with zeros
            feat = build_two_hand_feature(vis, left_lms, right_lms, NEED_XYZ)
            x = np.array(feat, dtype=np.float32).reshape(1, -1)
            probs = model.predict(x, verbose=0)[0]
            cls_idx = int(np.argmax(probs))
            pred_label = alphabet[cls_idx]
            pred_conf  = float(probs[cls_idx])

        else:
            # Model expects single-hand features; if both visible, pick higher-confidence hand
            best_label, best_conf = "-", 0.0

            if left_lms is not None:
                fL = hand_feature(vis, left_lms, NEED_XYZ)
                xL = np.array(fL, dtype=np.float32).reshape(1, -1)
                pL = model.predict(xL, verbose=0)[0]
                idxL = int(np.argmax(pL)); confL = float(pL[idxL]); labL = alphabet[idxL]
                best_label, best_conf = labL, confL

            if right_lms is not None:
                fR = hand_feature(vis, right_lms, NEED_XYZ)
                xR = np.array(fR, dtype=np.float32).reshape(1, -1)
                pR = model.predict(xR, verbose=0)[0]
                idxR = int(np.argmax(pR)); confR = float(pR[idxR]); labR = alphabet[idxR]
                if confR > best_conf:
                    best_label, best_conf = labR, confR

            # if no hands, stay as "-"
            pred_label, pred_conf = best_label, best_conf

        # -------- stability → commit char --------
        if pred_label != "-":
            chr_count[pred_label] += 1
            new_max = max(chr_count, key=chr_count.get)
            if new_max != current_max:
                current_max = new_max
                stable_start = now
            elif stable_start is None:
                stable_start = now

            held_ms = (now - stable_start) * 1000.0 if stable_start else 0.0
            since_last_commit = (now - last_commit_time) * 1000.0

            if current_max and held_ms >= STABLE_MS and since_last_commit >= DEBOUNCE_MS:
                current_word += current_max
                last_commit_time = now
                chr_count.clear()
                current_max = None
                stable_start = None
        else:
            # no valid pred this frame: soften stability (keep current_word)
            chr_count.clear()
            current_max = None
            stable_start = None

        # -------- keys --------
        key = cv2.waitKey(1) & 0xFF

        # Live speed switching
        if key == ord('1'): apply_speed("beginner")
        elif key == ord('2'): apply_speed("medium")
        elif key == ord('3'): apply_speed("fast")

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE → finalize word (single-call stub: corrected+translated)
            if current_word:
                raw = current_word
                def do_call():
                    global corrected_sentence, translated_sentence
                    corrected, translated = autocorrect_and_translate_single_call(raw, target_lang="Hindi")
                    if corrected:  corrected_sentence  += corrected  + " "
                    if translated: translated_sentence += translated + " "
                    print(f"[WORD] {raw} | [CORRECTED+] {corrected_sentence} | [TRANSLATED+] {translated_sentence}")
                async_call(do_call)

                current_word = ""
                chr_count.clear()
                current_max = None
                stable_start = None

        elif key == ord('u') or key == 8:  # undo/backspace char
            if current_word:
                current_word = current_word[:-1]
        elif key == ord('c'):  # clear everything (sentences + buffers)
            current_word = ""
            corrected_sentence = ""
            translated_sentence = ""
            chr_count.clear()
            current_max = None
            stable_start = None

        # -------- HUD (single combined prediction only) --------
        cv2.putText(vis, f"PRED: {pred_label} ({pred_conf:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 180, 60), 2)
        cv2.putText(vis, f"WORD: {current_word if current_word else '-'}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)
        cv2.putText(vis, f"CORRECTED: {corrected_sentence if corrected_sentence else '-'}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 0), 2)
        cv2.putText(vis, f"TRANSLATED: {translated_sentence if translated_sentence else '-'}",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 140, 0), 2)
        cv2.putText(vis, f"SPEED: {CURRENT_SPEED.upper()}  (STABLE={STABLE_MS}ms  DEBOUNCE={DEBOUNCE_MS}ms)",
                    (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)

        cv2.imshow("ISL — Single+Dual Hand (Unified Prediction)", vis)

cap.release()
cv2.destroyAllWindows()