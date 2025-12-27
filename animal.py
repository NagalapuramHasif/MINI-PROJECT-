import os
import numpy as np
import cv2
import tensorflow as tf
import warnings
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk

warnings.filterwarnings("ignore")

# ------------ CONSTANTS -----------------
# NOTE: Using RELATIVE paths. Files must be in the same directory as this script.
MODEL_PATH = "Best_Animal_Classifier.h5" 
DATA_DIR = "data/" 

# -------------------- LOAD MODEL --------------------------
model = None
CLASS_NAMES = ["Error: Model not loaded"]

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    CLASS_NAMES = sorted(os.listdir(DATA_DIR))
    print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")
except Exception as e:
    error_message = (
        f"Failed to load model or classes.\n\n"
        f"Ensure '{MODEL_PATH}' and '{DATA_DIR}' exist in the same folder.\n"
        f"Error: {e}"
    )
    messagebox.showerror("Model Load Error", error_message)

# -------------------- IMAGE PREPROCESS --------------------
def preprocess_image(image_path):
    """Loads, processes, and prepares an image for model prediction."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path.")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img.astype(np.float32))
    
    return np.expand_dims(img, axis=0)

# ---------------- PREDICTION FUNCTION ------------------------
def predict_image(image_path):
    """Makes a prediction using the loaded model."""
    if model is None:
        return "Model Unavailable", 0.0
        
    try:
        img_array = preprocess_image(image_path)
        preds = model.predict(img_array, verbose=0)
        
        class_id = np.argmax(preds[0])
        confidence = preds[0][class_id] * 100
        
        return CLASS_NAMES[class_id], confidence
        
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))
        return None, None
    
# ------------------ GUI CALLBACKS ----------------------------
def browse_image():
    """Handles image browsing, displays the image, and triggers prediction."""
    file_path = filedialog.askopenfilename(
        title=" Select Animal or Insect Image",
        filetypes=[(" Image files", "*.jpg *.jpeg *.png")]
    )
    
    if file_path:
        image = Image.open(file_path)
        image = image.resize((350, 350), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        
        image_label.config(image=photo, text="") 
        image_label.image = photo 
        
        predicted_class, confidence = predict_image(file_path)
        
        if predicted_class and confidence is not None and confidence > 0.0:
            result_label.config(
                text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%",
                fg="green"
            )
        elif predicted_class == "Model Unavailable": 
             result_label.config(
                 text="Error: Model file is missing or failed to load. Check console for path details.", 
                 fg="red"
             )
        elif predicted_class is not None:
             result_label.config(
                 text="Prediction Failed. Check image format.",
                 fg="red"
             )


# --------------------- GUI SETUP -----------------------
root = Tk()
root.title("General Animal  Classifier")
root.geometry("500x700") 

Label(root, text="General Animal  Classifier", font=("Arial", 20, "bold")).pack(pady=20)

image_label = Label(root, text="Image Preview", relief="sunken") 
image_label.pack(pady=10, padx=10)

browse_btn = Button(root, text="Select Animal Image", command=browse_image, font=("Arial", 14), bg="#90CAF9")
browse_btn.pack(pady=20)

result_label = Label(root, text="Select an image to classify the animal .", font=("Arial", 16), fg="blue")
result_label.pack(pady=10)

root.mainloop()