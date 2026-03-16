import cv2
import os
import sqlite3
import urllib.request
import numpy as np
import logging
from datetime import datetime
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("FaceEngine")


class AdvancedFaceEngine:
    def __init__(self, db_path="db", db_name="security_log.db"):
        self.db_path, self.db_name = db_path, db_name
        self.model_name, self.threshold = "ArcFace", 0.65

        os.makedirs(self.db_path, exist_ok=True)
        self._init_db()

        # Download face landmark model on first run (~3.8 MB)
        if not os.path.exists(_MODEL_PATH):
            logger.info("Downloading face landmarker model (~3.8 MB)...")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            logger.info("Model downloaded to %s", _MODEL_PATH)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._blink_counter = 0
        self._prev_nose_y = None

    def _init_db(self):
        conn = sqlite3.connect(self.db_name)
        conn.execute('CREATE TABLE IF NOT EXISTS access_logs (id INTEGER PRIMARY KEY, timestamp TEXT, name TEXT, confidence REAL)')
        conn.close()

    @staticmethod
    def _get_blendshape(blendshapes, name):
        for c in blendshapes:
            if c.category_name == name:
                return c.score
        return 0.0

    def check_liveness(self, frame_bgr):
        """
        Returns True when any natural facial movement is detected:
          • blink      – eyeBlinkLeft or eyeBlinkRight > 0.45 for ≥2 frames then opens
          • mouth open – jawOpen > 0.25
          • head nod   – nose-tip y-coordinate shifts > 0.018 (normalized) between frames
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        if not result.face_landmarks or not result.face_blendshapes:
            self._blink_counter = 0
            return False

        bs = result.face_blendshapes[0]

        # --- Blink via eyeBlink blendshapes ---
        blink_score = max(
            self._get_blendshape(bs, "eyeBlinkLeft"),
            self._get_blendshape(bs, "eyeBlinkRight"),
        )
        if blink_score > 0.45:
            self._blink_counter += 1
        else:
            if self._blink_counter >= 2:
                self._blink_counter = 0
                return True
            self._blink_counter = 0

        # --- Mouth open via jawOpen blendshape ---
        if self._get_blendshape(bs, "jawOpen") > 0.25:
            return True

        # --- Head nod via nose-tip y movement ---
        nose_y = result.face_landmarks[0][1].y
        if self._prev_nose_y is not None and abs(nose_y - self._prev_nose_y) > 0.018:
            self._prev_nose_y = nose_y
            return True
        self._prev_nose_y = nose_y

        return False

    def log_to_db(self, name, conf):
        try:
            conn = sqlite3.connect(self.db_name)
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute('INSERT INTO access_logs (timestamp, name, confidence) VALUES (?,?,?)', (t, name, conf))
            conn.commit()
            conn.close()
        except: pass

    def identify(self, frame):
        if not os.listdir(self.db_path): return {"status": "error", "message": "db empty"}
        try:
            dfs = DeepFace.find(img_path=frame, db_path=self.db_path, model_name=self.model_name, enforce_detection=False, silent=True)
            if dfs and not dfs[0].empty:
                best = dfs[0].iloc[0]
                dist = best['distance']
                if dist <= self.threshold:
                    name = os.path.splitext(os.path.basename(best['identity']))[0]
                    conf = round((1 - (dist / self.threshold)) * 100, 1)
                    self.log_to_db(name, conf)
                    return {"status": "success", "name": name, "confidence": conf}
            return {"status": "unknown"}
        except: return {"status": "error"}

    def register_new_user(self, frame, username):
        cv2.imwrite(os.path.join(self.db_path, f"{username}.jpg"), frame)
        cache = os.path.join(self.db_path, f"representations_{self.model_name.lower()}.pkl")
        if os.path.exists(cache): os.remove(cache)
        return True, "success"