import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import json
from datetime import datetime
import cv2
from PIL import Image, ImageTk


# Configuration Class
class Config:
    IMG_SIZE = (32, 32)
    IMG_CHANNELS = 3
    NUM_CLASSES = 10
    BATCH_SIZE = 64
    EPOCHS = 50
    VALIDATION_SPLIT = 0.1
    CHECKPOINT_PATH = "checkpoints/best_model.h5"
    FINAL_MODEL_PATH = "models/final_model.h5"
    LOG_DIR = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


# Load CIFAR-10 dataset
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, Config.NUM_CLASSES)
    y_test = to_categorical(y_test, Config.NUM_CLASSES)
    return (x_train, y_train), (x_test, y_test)


# Build Simple CNN
def build_simple_cnn(input_size=(32, 32, 3)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_size),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Build Transfer Learning Model
def build_transfer_model(model_name):
    input_shape = (32, 32, 3)

    if model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Unsupported model name")

    def resize_input(x):
        return tf.image.resize(x, (224, 224))

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Lambda(resize_input),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Callbacks
def get_callbacks(log_text):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    checkpoint = ModelCheckpoint(Config.CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True, mode='max')

    class LogCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            log_text.insert(tk.END, f"Epoch {epoch + 1} - Loss: {logs['loss']:.4f}, "
                                    f"Accuracy: {logs['accuracy']:.4f}, "
                                    f"Val Loss: {logs['val_loss']:.4f}, "
                                    f"Val Accuracy: {logs['val_accuracy']:.4f}\n")
            log_text.see(tk.END)

    return [early_stop, reduce_lr, checkpoint, LogCallback()]


# Plotting
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/training_history.png")
    plt.close()


def export_plot_as_pdf():
    doc = SimpleDocTemplate("training_report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Model Training Report", styles['Title']))
    story.append(Spacer(1, 12))

    if os.path.exists("plots/training_history.png"):
        story.append(Paragraph("Training History", styles['Heading2']))
        story.append(RLImage("plots/training_history.png", width=400, height=250))
        story.append(Spacer(1, 12))

    if os.path.exists("plots/confusion_matrix.png"):
        story.append(Paragraph("Confusion Matrix", styles['Heading2']))
        story.append(RLImage("plots/confusion_matrix.png", width=400, height=300))

    doc.build(story)
    messagebox.showinfo("Export", "PDF report saved as 'training_report.pdf'")


def plot_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Config.CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    plt.close()


def log_results(history, test_loss, test_acc):
    result = {
        "final_test_accuracy": float(test_acc),
        "final_test_loss": float(test_loss),
        "best_validation_accuracy": max(history.history['val_accuracy']),
        "best_training_accuracy": max(history.history['accuracy']),
    }
    with open("results.json", 'w') as f:
        json.dump(result, f, indent=4)


# GUI Application
class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Classifier with Webcam + Custom Prediction")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        self.create_widgets()

        # Load data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar10()
        val_split = int(Config.VALIDATION_SPLIT * len(self.x_train))
        self.x_val, self.y_val = self.x_train[:val_split], self.y_train[:val_split]
        self.x_train, self.y_train = self.x_train[val_split:], self.y_train[val_split:]
        self.model = None
        self.history = None

    def create_widgets(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)

        ttk.Label(control_frame, text="Select Model:").grid(row=0, column=0, padx=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=["Simple CNN", "VGG16", "ResNet50"]
        )
        self.model_combo.grid(row=0, column=1, padx=5)

        ttk.Button(control_frame, text="Train Model", command=self.train_model).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Evaluate Model", command=self.evaluate_model).grid(row=0, column=3, padx=5)

        image_frame = ttk.Frame(self.root)
        image_frame.pack(pady=10)

        self.canvas = tk.Canvas(image_frame, width=320, height=320)
        self.canvas.pack(side=tk.LEFT, padx=10)

        pred_frame = ttk.Frame(image_frame)
        pred_frame.pack(side=tk.LEFT)

        self.pred_label = ttk.Label(pred_frame, text="Prediction: N/A", font=("Arial", 14))
        self.pred_label.pack(pady=5)

        ttk.Button(pred_frame, text="Upload Image", command=self.upload_image).pack(pady=5)
        ttk.Button(pred_frame, text="Capture from Webcam", command=self.open_webcam).pack(pady=5)

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Plot Training History", command=self.plot_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Plot Confusion Matrix", command=self.plot_confusion).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Report as PDF", command=export_plot_as_pdf).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)

        self.log_text = scrolledtext.ScrolledText(self.root, height=10)
        self.log_text.pack(pady=10)

    def train_model(self):
        model_type = self.model_var.get()
        if not model_type:
            self.log_text.insert(tk.END, "Please select a model type.\n")
            return
        self.log_text.insert(tk.END, f"[INFO] Building {model_type}...\n")
        if model_type == "Simple CNN":
            self.model = build_simple_cnn()
        elif model_type == "VGG16":
            self.model = build_transfer_model("VGG16")
        elif model_type == "ResNet50":
            self.model = build_transfer_model("ResNet50")
        self.log_text.insert(tk.END, "[INFO] Starting Training...\n")
        callbacks = get_callbacks(self.log_text)
        history = self.model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks
        )
        self.history = history
        self.log_text.insert(tk.END, "[INFO] Training completed.\n")

    def evaluate_model(self):
        if self.model is None:
            self.log_text.insert(tk.END, "[ERROR] No model loaded or trained yet.\n")
            return
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.log_text.insert(tk.END, f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}\n")
        log_results(self.history, loss, accuracy)

    def plot_training(self):
        if hasattr(self, 'history'):
            plot_training_history(self.history)
            self.log_text.insert(tk.END, "[INFO] Training history plotted.\n")
        else:
            self.log_text.insert(tk.END, "[ERROR] No training history found.\n")

    def plot_confusion(self):
        if self.model is None:
            self.log_text.insert(tk.END, "[ERROR] No model loaded or trained yet.\n")
            return
        plot_confusion_matrix(self.model, self.x_test, self.y_test)
        self.log_text.insert(tk.END, "[INFO] Confusion matrix plotted.\n")

    def save_model(self):
        if self.model is None:
            self.log_text.insert(tk.END, "[ERROR] No model loaded or trained yet.\n")
            return
        path = filedialog.asksaveasfilename(defaultextension=".h5",
                                            filetypes=[("HDF5 Files", "*.h5")])
        if path:
            self.model.save(path)
            self.log_text.insert(tk.END, f"[INFO] Model saved to {path}\n")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("HDF5 Files", "*.h5")])
        if path:
            try:
                self.model = tf.keras.models.load_model(path)
                self.log_text.insert(tk.END, f"[INFO] Model loaded from {path}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load model:\n{e}")

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self._predict_custom_image(path)

    def open_webcam(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Press SPACE to Capture", frame)
            key = cv2.waitKey(1)
            if key == 32:
                temp_path = "temp_webcam.jpg"
                cv2.imwrite(temp_path, frame)
                cap.release()
                cv2.destroyAllWindows()
                self._predict_custom_image(temp_path)
                break

    def _predict_custom_image(self, image_path):
        try:
            img = load_img(image_path, target_size=(32, 32))
            img_arr = img_to_array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            pred = self.model.predict(img_arr)
            predicted_class = Config.CLASS_NAMES[np.argmax(pred)]
            self.pred_label.config(text=f"Prediction: {predicted_class}")

            pil_img = Image.open(image_path).resize((320, 320))
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        except Exception as e:
            messagebox.showerror("Error", str(e))


# Main Execution
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()