import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

def build_unet_efficientnet(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # ENCODER (Pre-trained EfficientNetB0)
    encoder = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
    encoder.trainable = False 
    
    s1 = encoder.get_layer("input_layer").output                   # Corrected
    s2 = encoder.get_layer("block2a_expand_activation").output    
    s3 = encoder.get_layer("block3a_expand_activation").output    
    s4 = encoder.get_layer("block4a_expand_activation").output    
    s5 = encoder.get_layer("block6a_expand_activation").output    
    
    b1 = encoder.get_layer("top_activation").output               
    
    # DECODER
    def decoder_block(input_tensor, skip_tensor, num_filters):
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_tensor)
        x = layers.Concatenate()([x, skip_tensor])
        x = layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
        return x

    d1 = decoder_block(b1, s5, 512)   
    d2 = decoder_block(d1, s4, 256)   
    d3 = decoder_block(d2, s3, 128)   
    d4 = decoder_block(d3, s2, 64)    
    
    x = layers.Conv2DTranspose(32, (2, 2), strides=2, padding="same")(d4)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    
    # OUTPUT
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Sentinel_UNet")
    return model