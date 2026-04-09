import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def evaluate_and_visualize(model, dataset, num_samples=3):
    for images, masks in dataset.take(1):
        preds = model.predict(images)
        preds_binary = (preds > 0.5).astype(np.float32)

        plt.figure(figsize=(15, 5 * num_samples))
        for i in range(num_samples):
            # Reverse preprocessing for display
            img_display = images[i].numpy()
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
            
            mask_display = masks[i].numpy().squeeze()
            pred_display = preds_binary[i].squeeze()
            
            plt.subplot(num_samples, 3, i * 3 + 1)
            plt.title("Input Image")
            plt.imshow(img_display)
            plt.axis("off")
            
            plt.subplot(num_samples, 3, i * 3 + 2)
            plt.title("Ground Truth Mask")
            plt.imshow(mask_display, cmap="gray")
            plt.axis("off")
            
            plt.subplot(num_samples, 3, i * 3 + 3)
            plt.title("Predicted Mask")
            plt.imshow(pred_display, cmap="gray")
            plt.axis("off")
            
        plt.tight_layout()
        plt.show()