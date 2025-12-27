import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import warnings

warnings.filterwarnings("ignore")

# ------------------ CONSTANTS ------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'data/'      
LEARNING_RATE = 0.001
EPOCHS = 1            
SEED = 123
MODEL_FILENAME = "Best_Animal_Classifier.h5"

# ------------------ LOAD DATA ------------------
print("Loading data...")
try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
except Exception as e:
    print(f"\nFATAL ERROR: Failed to load data from '{DATA_DIR}'.")
    print(e)
    exit()

CLASS_NAMES = np.array(train_ds.class_names)
NUM_CLASSES = len(CLASS_NAMES)
print(f"Detected {NUM_CLASSES} animal classes: {CLASS_NAMES}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ------------------ DATA AUGMENTATION ------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.5),
    tf.keras.layers.RandomZoom(0.3),
], name="data_augmentation")

# ------------------ MODEL BUILD ------------------
base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------ TRAIN MODEL ------------------
print("\nStarting model training...")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_FILENAME,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        monitor='val_accuracy',
        mode='max'
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"\n✅ Training finished. Best model saved as '{MODEL_FILENAME}'")

# ------------------ PLOT TRAINING & VALIDATION LOSS ------------------
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs_range, loss, 'bo-', label='Training Loss')
plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# --------- POST-TRAINING EVALUATION METRICS -------------

print("\n🔍 Calculating detailed evaluation metrics...")

model.load_weights(MODEL_FILENAME)

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

unique_labels = np.unique(y_true)
filtered_class_names = CLASS_NAMES[unique_labels]

# --- Classification Report ---
report = classification_report(
    y_true,
    y_pred,
    target_names=filtered_class_names,
    digits=4,
    zero_division=0
)
print("\n📊 Classification Report:")
print(report)

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm, filtered_class_names)

# --- Overall Accuracy ---
acc = np.mean(y_true == y_pred)
print(f"\n✅ Final Validation Accuracy: {acc:.4f}")
