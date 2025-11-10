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
    "beginner": {"STABLE_MS": 1200, "DEBOUNCE_MS": int(1200 * 0.30)},  # 360ms
    "medium":   {"STABLE_MS": 1000, "DEBOUNCE_MS": int(1000 * 0.30)},  # 300ms
    "fast":     {"STABLE_MS": 800,  "DEBOUNCE_MS": int(800 * 0.30)},   # 240ms
}

CURRENT_SPEED = "medium"  # default
STABLE_MS = SPEED_PRESETS[CURRENT_SPEED]["STABLE_MS"]
DEBOUNCE_MS = SPEED_PRESETS[CURRENT_SPEED]["DEBOUNCE_MS"]
SHOW_MIRROR = True
COMMIT_SOURCE = "maxconf"  # "maxconf" | "right" | "left"

def apply_speed(preset_name: str):
    global CURRENT_SPEED, STABLE_MS, DEBOUNCE_MS
    if preset_name not in SPEED_PRESETS:
        return
    CURRENT_SPEED = preset_name
    STABLE_MS = SPEED_PRESETS[preset_name]["STABLE_MS"]
    DEBOUNCE_MS = SPEED_PRESETS[preset_name]["DEBOUNCE_MS"]
    print(f"[SPEED] Set to {CURRENT_SPEED.upper()} → STABLE_MS={STABLE_MS}ms, DEBOUNCE_MS={DEBOUNCE_MS}ms")

# Which hand's char to commit:
#   "maxconf" = whichever hand has higher softmax prob this frame
#   "right"   = only right hand
#   "left"    = only left hand
COMMIT_SOURCE = "maxconf"

# =================== MODEL & MP ===================
model = keras.models.load_model("model.h5")
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = list("123456789") + list(string.ascii_uppercase)

# =================== HELPERS ===================
def calc_landmark_list(image, landmarks):
    h, w = image.shape[0], image.shape[1]
    pts = []
    for lm in landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        pts.append([x, y])
    return pts

def pre_process_landmark(landmark_list):
    tmp = copy.deepcopy(landmark_list)
    base_x, base_y = tmp[0][0], tmp[0][1]
    for i in range(len(tmp)):
        tmp[i][0] -= base_x
        tmp[i][1] -= base_y
    tmp = list(itertools.chain.from_iterable(tmp))
    maxv = max(map(abs, tmp)) or 1.0
    return [v / maxv for v in tmp]

# ======== STUB: single-call autocorrect+translate (replace later) ========
def autocorrect_and_translate_single_call(word, target_lang="Hindi"):
    # Placeholder: real integration will call your backend once and return both fields.
    corrected = word  # TODO: replace with Gemini corrected
    translated = word # TODO: replace with Gemini translation
    return corrected, translated

def async_call(fn, *args, **kwargs):
    t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    t.start()

# =================== STATE ===================
chr_count = defaultdict(int)
current_max = None
stable_start = None
last_commit_time = 0.0

current_word = ""         # raw characters (committed via stability)
corrected_word = "-"      # filled on SPACE via single-call stub
translated_word = "-"     # filled on SPACE via single-call stub
corrected_sentence = ""
translated_sentence = ""

# track which hand we're currently using to stabilize (so hand switches reset timer)
active_hand_for_stability = None  # "Right" / "Left"

# =================== VIDEO LOOP ===================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Camera not available")

# -------- choose speed once (console) --------
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

        # ---------- per-hand predictions ----------
        hand_preds = []  # list of dicts: {label, conf, handedness, box_center}
        if results.multi_hand_landmarks:
            for hand_lms, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # draw
                mp_drawing.draw_landmarks(
                    vis, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                # handedness label
                handed_label = handness.classification[0].label  # "Right" / "Left"

                lm_list = calc_landmark_list(vis, hand_lms)
                pre = pre_process_landmark(lm_list)
                df = pd.DataFrame(pre).transpose()

                probs = model.predict(df, verbose=0)[0]
                cls_idx = int(np.argmax(probs))
                label = alphabet[cls_idx]
                conf = float(probs[cls_idx])  # softmax confidence

                # approx position for HUD
                cx = int(np.mean([p[0] for p in lm_list]))
                cy = int(np.mean([p[1] for p in lm_list]))
                hand_preds.append({
                    "label": label,
                    "conf": conf,
                    "handed": handed_label,
                    "pos": (cx, cy)
                })

            # sort for deterministic drawing (Right on top line)
            hand_preds.sort(key=lambda d: d["handed"], reverse=True)

            # show BOTH hands' predictions
            y0 = 30
            for i, hp in enumerate(hand_preds):
                txt = f"{hp['handed']}: {hp['label']} ({hp['conf']:.2f})"
                cv2.putText(vis, txt, (20, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # pick which hand contributes to stability/commit
            if COMMIT_SOURCE == "right":
                chosen = next((hp for hp in hand_preds if hp["handed"] == "Right"), None)
            elif COMMIT_SOURCE == "left":
                chosen = next((hp for hp in hand_preds if hp["handed"] == "Left"), None)
            else:  # "maxconf"
                chosen = max(hand_preds, key=lambda d: d["conf"], default=None)

            if chosen:
                # if hand switched, reset stability timer & counts
                if active_hand_for_stability != chosen["handed"]:
                    active_hand_for_stability = chosen["handed"]
                    chr_count.clear()
                    current_max = None
                    stable_start = None

                # stability logic
                chr_count[chosen["label"]] += 1
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
            # no hands → reset short-term stability (keep current_word)
            chr_count.clear()
            current_max = None
            stable_start = None
            active_hand_for_stability = None

        # ---------- KEYS ----------
        key = cv2.waitKey(1) & 0xFF

        # Live speed switching
        if key == ord('1'):
            apply_speed("beginner")
        elif key == ord('2'):
            apply_speed("medium")
        elif key == ord('3'):
            apply_speed("fast")

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE → finalize word, single-call stub for both corrected+translated
            if current_word:
                raw = current_word
                def do_call():
                    global corrected_word, translated_word, corrected_sentence, translated_sentence
                    corrected, translated = autocorrect_and_translate_single_call(raw, target_lang="Hindi")
                    corrected_word = corrected if corrected else "-"
                    translated_word = translated if translated else "-"
                    # APPEND to running sentences
                    if corrected:
                        corrected_sentence += corrected + " "
                    if translated:
                        translated_sentence += translated + " "
                    print(f"[WORD] {raw} | [CORRECTED+] {corrected_sentence} | [TRANSLATED+] {translated_sentence}")
                async_call(do_call)

                current_word = ""
                chr_count.clear()
                current_max = None
                stable_start = None
                active_hand_for_stability = None

        elif key == ord('u'):  # undo last char
            if current_word:
                current_word = current_word[:-1]
        # Backspace key (keycode 8)
        elif key == 8:
            if current_word:
                current_word = current_word[:-1]
        elif key == ord('c'):  # clear everything (not the last corrected/translated results)
            current_word = ""
            chr_count.clear()
            current_max = None
            stable_start = None
            active_hand_for_stability = None
            corrected_sentence = ""
            translated_sentence = ""

        # ---------- HUD ----------
        cv2.putText(vis, f"COMMIT_FROM: {COMMIT_SOURCE.upper()}", (20, h-110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(vis, f"WORD: {current_word if current_word else '-'}", (20, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,180), 2)
        # cv2.putText(vis, f"CORRECTED: {corrected_word}", (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,220,0), 2)
        # cv2.putText(vis, f"TRANSLATED: {translated_word}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,140,0), 2)
        cv2.putText(vis, f"CORRECTED: {corrected_sentence if corrected_sentence else '-'}", (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,220,0), 2)
        cv2.putText(vis, f"TRANSLATED: {translated_sentence if translated_sentence else '-'}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,140,0), 2)
        cv2.putText(vis, f"SPEED: {CURRENT_SPEED.upper()}  (STABLE={STABLE_MS}ms  DEBOUNCE={DEBOUNCE_MS}ms)",(20, h-140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)

        cv2.imshow("ISL - Two-Hand Detection + Word/Corrected/Translated", vis)

cap.release()
cv2.destroyAllWindows()