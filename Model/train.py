import tensorflow as tf
from dataset import create_dataset
from model import build_unet_efficientnet
from evaluate import dice_coefficient, evaluate_and_visualize

# 1. Define Paths (Update these based on where your data lives)
TRAIN_IMG_DIR = 'data/train/images'
TRAIN_MASK_DIR = 'data/train/masks'
VAL_IMG_DIR = 'data/train/images'  # Using train data for validation for now
VAL_MASK_DIR = 'data/train/masks' # Using train data for validation for now

# 2. Create Datasets
print("Loading datasets...")
train_dataset = create_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
val_dataset = create_dataset(VAL_IMG_DIR, VAL_MASK_DIR)

# Handle case where val_dataset might be empty if the directories don't exist
if tf.data.experimental.cardinality(val_dataset) == 0:
    print("Validation dataset is empty. Using training dataset for validation.")
    val_dataset = train_dataset

# 3. Build & Compile Model
print("Building model...")
model = build_unet_efficientnet(input_shape=(224, 224, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", dice_coefficient, tf.keras.metrics.MeanIoU(num_classes=2, name="iou")]
)

# 4. Train Model
print("Starting training...")
EPOCHS = 10

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# 5. Save Model
model.save("unet_efficientnet_deepfake_detector.h5")

# 6. Evaluate & Visualize
print("Visualizing results...")
evaluate_and_visualize(model, val_dataset)