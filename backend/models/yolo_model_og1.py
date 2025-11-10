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

# ------------------ CONFIG ------------------
STABLE_MS = 1000            # hold time to accept a character
DEBOUNCE_MS = 250          # avoid immediate repeat commit
MIN_DET_CONF = 0.50        # not used (mediapipe hands only), keep for future
SHOW_MIRROR = True         # mirror preview only

# ------------------ MODEL & MP ------------------
model = keras.models.load_model("model.h5")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = list("123456789") + list(string.ascii_uppercase)

# ------------------ HELPERS ------------------
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

# ------------------ PARALLEL STUBS (no integration) ------------------
def send_word_to_autocorrect_and_translate(word):
    # Placeholder: do your Gemini calls here (non-blocking)
    print(f"[STUB] send to Gemini → word='{word}' (autocorrect + translate)")

def async_call(fn, *args, **kwargs):
    t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    t.start()

# ------------------ STATE ------------------
chr_count = defaultdict(int)
current_max = None
stable_start = None
last_committed_char = None
last_commit_time = 0.0

current_word = ""     # building word (characters committed by stability)
full_sentence = ""    # final output (words + spaces)

# ------------------ VIDEO LOOP ------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera not available")

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Ignoring empty camera frame.")
            continue

        if SHOW_MIRROR:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)

        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        vis.flags.writeable = True

        now = time.monotonic()

        if results.multi_hand_landmarks:
            # use first detected hand (can expand later)
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                vis, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            lm_list = calc_landmark_list(vis, hand_landmarks)
            pre = pre_process_landmark(lm_list)
            df = pd.DataFrame(pre).transpose()

            preds = model.predict(df, verbose=0)
            cls_idx = int(np.argmax(preds, axis=1)[0])
            label = alphabet[cls_idx]

            # count & stability logic
            chr_count[label] += 1
            # find new max char
            new_max = max(chr_count, key=chr_count.get)
            if new_max != current_max:
                current_max = new_max
                stable_start = now  # reset stability timer
            elif stable_start is None:
                stable_start = now

            # commit if held steady long enough and not a rapid duplicate
            held_ms = (now - stable_start) * 1000.0 if stable_start else 0
            since_last_commit = (now - last_commit_time) * 1000.0
            if current_max and held_ms >= STABLE_MS and since_last_commit >= DEBOUNCE_MS:
                # commit char
                current_word += current_max
                last_committed_char = current_max
                last_commit_time = now
                # reset counters for next char
                chr_count.clear()
                current_max = None
                stable_start = None

            # HUD
            cv2.putText(vis, f"PRED: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 255), 2)
        else:
            # no hand → soften counts & timers (prevents stale commits)
            chr_count.clear()
            current_max = None
            stable_start = None

        # --- KEYS ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE → finalize current word, enqueue for Gemini
            if current_word:
                word = current_word
                full_sentence += (word + " ")
                print(f"[COMMIT WORD] '{word}'  | Sentence: '{full_sentence}'")
                async_call(send_word_to_autocorrect_and_translate, word)
                current_word = ""
                chr_count.clear()
                current_max = None
                stable_start = None
        elif key == ord('c'):  # Clear sentence & word
            current_word = ""
            full_sentence = ""
            chr_count.clear()
            current_max = None
            stable_start = None
            print("[CLEAR] buffers reset")
        elif key == ord('u'):  # Undo last char in current word
            if current_word:
                current_word = current_word[:-1]
        elif key == ord('b'):  # Backspace sentence (remove trailing space or last word)
            if full_sentence.endswith(" "):
                full_sentence = full_sentence[:-1]
            # remove last contiguous non-space chunk
            full_sentence = full_sentence.rsplit(" ", 1)[0] + (" " if full_sentence.endswith(" ") else "")

        # --- OVERLAY TEXT ---
        cv2.putText(vis, f"WORD: {current_word}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)
        cv2.putText(vis, f"SENTENCE: {full_sentence}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 0), 2)
        cv2.putText(vis, "SPACE=commit word | C=clear | U=undo char | B=backspace word | ESC=quit",
                    (20, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("ISL - Stable Character Commit", vis)

cap.release()
cv2.destroyAllWindows()