import customtkinter as ctk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import threading

# --- 1. SETUP UI THEME ---
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# --- 2. THE MALWARE DETECTION LOGIC ---
class_names = [
    'Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 
    'Benign', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 
    'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 
    'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 
    'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A'
]

def file_to_image(file_path, target_size=(224, 224)):
    if file_path.lower().endswith('.png'):
        img = Image.open(file_path).convert('RGB')
    else:
        with open(file_path, 'rb') as f:
            content = f.read()
        d = np.frombuffer(content, dtype=np.uint8)
        
        kb = len(d) / 1024
        if kb < 10: width = 32
        elif kb < 30: width = 64
        elif kb < 60: width = 128
        elif kb < 100: width = 256
        elif kb < 200: width = 384
        elif kb < 500: width = 512
        elif kb < 1000: width = 768
        else: width = 1024
        
        height = len(d) // width
        if height == 0: return None
        img_array = d[:width * height].reshape((height, width))
        img = Image.fromarray(img_array).convert('RGB')

    img = img.resize(target_size, Image.BILINEAR)
    return np.array(img).reshape(1, 224, 224, 3).astype('float32')

# --- 3. THE DESKTOP APPLICATION ---
class MalwareScannerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ResNet-50 Malware Scanner")
        self.geometry("600x450")
        self.selected_file = None
        self.model = None

        # --- UI LAYOUT ---
        # Title
        self.title_label = ctk.CTkLabel(self, text="Deep Learning Malware Scanner", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(20, 5))

        self.subtitle_label = ctk.CTkLabel(self, text="Select a binary (.exe) or image (.png) to analyze", text_color="gray")
        self.subtitle_label.pack(pady=(0, 20))

        # Select Button
        self.select_btn = ctk.CTkButton(self, text="Select File", command=self.browse_file, width=200, height=40)
        self.select_btn.pack(pady=10)

        # File Path Display
        self.file_label = ctk.CTkLabel(self, text="No file selected", font=ctk.CTkFont(size=12))
        self.file_label.pack(pady=5)

        # Analyze Button
        self.analyze_btn = ctk.CTkButton(self, text="Analyze File", command=self.start_analysis, width=200, height=40, state="disabled")
        self.analyze_btn.pack(pady=20)

        # Results Frame (Box to hold the results)
        self.result_frame = ctk.CTkFrame(self, width=500, height=120, corner_radius=10)
        self.result_frame.pack(pady=10, fill="x", padx=40)
        self.result_frame.pack_propagate(False)

        self.result_title = ctk.CTkLabel(self.result_frame, text="Awaiting File...", font=ctk.CTkFont(size=18, weight="bold"))
        self.result_title.pack(pady=(20, 5))

        self.result_details = ctk.CTkLabel(self.result_frame, text="", font=ctk.CTkFont(size=14))
        self.result_details.pack(pady=5)

        # Load Model in the background so the UI doesn't freeze on startup
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        self.result_title.configure(text="Loading Model...", text_color="orange")
        try:
            # UPDATE THIS PATH TO YOUR MODEL
            model_path = './models/malware_resnet50.h5'
            self.model = tf.keras.models.load_model(model_path)
            self.result_title.configure(text="System Ready", text_color="green")
        except Exception as e:
            self.result_title.configure(text="Error Loading Model", text_color="red")
            print(e)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a file to scan",
            filetypes=(("Executables", "*.exe"), ("PNG Images", "*.png"), ("All Files", "*.*"))
        )
        if file_path:
            self.selected_file = file_path
            self.file_label.configure(text=f"...{file_path[-40:]}") # Show end of path
            if self.model is not None:
                self.analyze_btn.configure(state="normal") # Enable analyze button

    def start_analysis(self):
        # Update UI to show processing state
        self.analyze_btn.configure(state="disabled")
        self.result_title.configure(text="Scanning...", text_color="orange")
        self.result_details.configure(text="")
        
        # Run inference in a separate thread to keep UI responsive
        threading.Thread(target=self.run_inference, daemon=True).start()

    def run_inference(self):
        try:
            input_data = file_to_image(self.selected_file)
            if input_data is None:
                self.update_ui("Error", "File could not be processed.", "red")
                return

            preds = self.model.predict(input_data, verbose=0)
            class_idx = np.argmax(preds)
            prob_val = np.max(preds)
            confidence_pct = prob_val * 100
            label = class_names[class_idx]

            THRESHOLD = 0.90 

            if prob_val < THRESHOLD:
                self.update_ui(f"⚠️ INCONCLUSIVE. Could be: {label}", f"Confidence ({confidence_pct:.2f}%) below safety threshold.", "orange")
            elif label == "Benign":
                self.update_ui("✅ SAFE: Benign", f"Confidence: {confidence_pct:.2f}%", "green")
            else:
                self.update_ui(f"🚨 MALWARE: {label}", f"Confidence: {confidence_pct:.2f}%", "red")

        except Exception as e:
            self.update_ui("Error", str(e), "red")
        finally:
            self.analyze_btn.configure(state="normal")

    def update_ui(self, title, details, color):
        # Thread-safe UI update
        self.result_title.configure(text=title, text_color=color)
        self.result_details.configure(text=details)

# --- 4. RUN THE APP ---
if __name__ == "__main__":
    app = MalwareScannerApp()
    app.mainloop()