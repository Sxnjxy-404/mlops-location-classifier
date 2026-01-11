import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------
# Config
# -------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
DATA_DIR = "data/train"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "location_model.keras")
CLASS_PATH = os.path.join(MODEL_DIR, "classes.json")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Reproducibility
tf.random.set_seed(42)

# -------------------
# Data
# -------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)

# -------------------
# Model
# -------------------
base = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)
base.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
output = tf.keras.layers.Dense(train_data.num_classes, activation="softmax")(x)
model = tf.keras.Model(base.input, output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------
# Training
# -------------------
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# -------------------
# Save model + labels
# -------------------
model.save(MODEL_PATH)

with open(CLASS_PATH, "w") as f:
    json.dump(train_data.class_indices, f, indent=2)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Classes saved to {CLASS_PATH}")
