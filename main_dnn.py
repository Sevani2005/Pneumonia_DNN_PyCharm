import os
import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

# ===== PATHS =====
# Default local path (good for Git)
base_path = "data" 

# Fallback to your specific local path if 'data' folder doesn't exist
if not os.path.exists(base_path):
    base_path = r"C:\Users\sevpr\Downloads\archive (1)\chest_xray"

data_dir = base_path

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

# ===== SAVE MODEL =====
model.save("pneumonia_model.keras")
print("Model saved as pneumonia_model.keras")

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
# plt.show() # Commented out for automated environments if necessary
plt.savefig("accuracy_plot.png")
print("Accuracy plot saved as accuracy_plot.png")

# ===== PREDICT ONE IMAGE =====
def predict_single_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, h, w, 3) [web:765][web:751]

    prob = model.predict(img_array)[0][0]         # sigmoid output between 0 and 1 [web:765][web:769]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    print(f"Prediction: {label}  (prob={prob:.3f})")

# change this path for ANY image you want to test
# Example using relative path:
example_path = os.path.join(test_dir, "NORMAL", os.listdir(os.path.join(test_dir, "NORMAL"))[0]) if os.path.exists(test_dir) else "path_to_test_image.jpg"
predict_single_image(example_path)
