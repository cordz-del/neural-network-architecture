import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Define a custom residual block
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    # First convolutional layer
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Second convolutional layer
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    # Adjust shortcut dimensions if necessary
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    # Combine the shortcut with the main path
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Build the CNN with residual blocks using the Functional API
input_tensor = Input(shape=(224, 224, 3))
x = Conv2D(64, kernel_size=7, strides=2, padding='same')(input_tensor)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Stack a couple of residual blocks
x = residual_block(x, filters=64)
x = residual_block(x, filters=64)

# Global pooling and dense output layer for classification
x = GlobalAveragePooling2D()(x)
output_tensor = Dense(10, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
