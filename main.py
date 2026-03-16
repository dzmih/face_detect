import threading
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from face_engine import AdvancedFaceEngine


class App:
    def __init__(self, window, title):
        self.window = window
        self.window.title(title)

        self.engine = AdvancedFaceEngine()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.vid = cv2.VideoCapture(1)
        if not self.vid.isOpened():
            self.vid = cv2.VideoCapture(0)

        if not self.vid.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam (index 0 or 1).")
            self.window.destroy()
            return

        self.canvas = tk.Canvas(window, width=640, height=480, bg="black")
        self.canvas.pack()

        self.status_var = tk.StringVar(value="status: ready")
        ttk.Label(window, textvariable=self.status_var, font=("Arial", 12)).pack(pady=5)

        self.reg_frame = ttk.Frame(window)
        self.reg_frame.pack(pady=5)

        ttk.Label(self.reg_frame, text="name:").grid(row=0, column=0)
        self.name_entry = ttk.Entry(self.reg_frame)
        self.name_entry.grid(row=0, column=1, padx=5)
        ttk.Button(self.reg_frame, text="register me", command=self.save_user).grid(
            row=0, column=2
        )

        self.is_recognizing = False
        self.current_name = "blink, open mouth or nod..."
        self.is_live = False
        self.live_timer = 0
        self.last_frame = None

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.last_frame = frame.copy()
            frame_display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if self.is_live:
                self.live_timer -= 1
                if self.live_timer <= 0:
                    self.is_live = False
                    self.current_name = "blink, open mouth or nod..."

            if len(faces) > 0 and not self.is_live and self.engine.check_liveness(frame):
                self.is_live = True
                self.live_timer = 150

            for (x, y, w, h) in faces:
                color = (0, 255, 0) if self.is_live else (0, 0, 255)
                cv2.rectangle(frame_display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame_display,
                    f"id: {self.current_name}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            if len(faces) > 0 and not self.is_live:
                cv2.putText(
                    frame_display,
                    "BLINK / OPEN MOUTH / NOD",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            if len(faces) > 0 and self.is_live and not self.is_recognizing:
                self.is_recognizing = True
                threading.Thread(target=self.run_id, args=(frame.copy(),), daemon=True).start()

            img = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update)

    def run_id(self, frame):
        try:
            result = self.engine.identify(frame)
            status = result.get("status")

            if status == "success":
                name = result.get("name", "unknown")
                conf = result.get("confidence", 0)
                self.current_name = name
                self.status_var.set(f"status: access granted for {name} ({conf}%)")
            elif status == "unknown":
                self.current_name = "unknown"
                self.status_var.set("status: face not recognized")
            else:
                self.current_name = "error"
                msg = result.get("message", "identification error")
                self.status_var.set(f"status: {msg}")
        except Exception as exc:
            self.current_name = "error"
            self.status_var.set(f"status: runtime error - {exc}")
        finally:
            self.is_recognizing = False

    def save_user(self):
        username = self.name_entry.get().strip()
        if not username:
            messagebox.showwarning("Input", "Please enter a user name.")
            return

        if self.last_frame is None:
            messagebox.showwarning("Camera", "No camera frame available yet.")
            return

        ok, message = self.engine.register_new_user(self.last_frame, username)
        if ok:
            self.status_var.set(f"status: user '{username}' registered")
            self.name_entry.delete(0, tk.END)
        else:
            self.status_var.set(f"status: registration failed ({message})")

    def on_close(self):
        if hasattr(self, "vid") and self.vid and self.vid.isOpened():
            self.vid.release()
        self.window.destroy()


def main():
    root = tk.Tk()
    App(root, "Sentinel AI")
    root.mainloop()


if __name__ == "__main__":
    main()