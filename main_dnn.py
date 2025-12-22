import os
import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# ===== PATHS =====
data_dir = r"C:\Users\sevpr\Downloads\archive (1)\chest_xray"  # change ONLY if your chest_xray is elsewhere

img_size = (180, 180)
batch_size = 32

train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

print("Train folder exists:", os.path.exists(train_dir))
print("Val folder exists:", os.path.exists(val_dir))
print("Test folder exists:", os.path.exists(test_dir))

if not os.path.exists(train_dir):
    print("ERROR: train folder not found. Check data_dir path.")
    raise SystemExit

print("Train subfolders:", os.listdir(train_dir))

# ===== LOAD DATASETS =====
train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
)

val_ds = keras.utils.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="binary",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
)

test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="binary",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False,
)

# ===== PREFETCH =====
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# ===== BUILD MODEL (CNN) =====
model = models.Sequential([
    layers.Rescaling(1. / 255, input_shape=img_size + (3,)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),  # binary output
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ===== TRAIN =====
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

# ===== EVALUATE ON TEST SET =====
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

# ===== PLOT ACCURACY =====
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.show()

# ===== PREDICT ONE IMAGE =====
def predict_single_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, h, w, 3) [web:765][web:751]

    prob = model.predict(img_array)[0][0]         # sigmoid output between 0 and 1 [web:765][web:769]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    print(f"Prediction: {label}  (prob={prob:.3f})")

# change this path for ANY image you want to test
example_path = r"C:\Users\sevpr\Downloads\archive (1)\chest_xray\chest_xray\test\NORMAL\IM-0001-0001.jpeg"
predict_single_image(example_path)
