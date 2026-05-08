# Deep Learning: Convolutional Neural Networks (CNNs)

## Overview
**Convolutional Neural Networks (CNNs)** are specialized neural networks designed for processing grid-like data (images, videos).
- Inspired by visual cortex of animals
- Use convolutional layers to detect patterns
- Share weights to reduce parameters
- State-of-the-art for computer vision tasks
- Also used for text, time-series, and other 1D/2D data

---

## Why CNNs for Images?

### Problems with Fully Connected Networks
```
Input Image: 224×224×3 = 150,528 pixels
Hidden Layer 1: 1,000 neurons = 150,528,000 parameters

Problems:
- Huge number of parameters
- Computationally expensive
- Ignores spatial structure
- Poor generalization
```

### CNN Solution
```
Convolution Layer: Learn local patterns
Pooling Layer: Reduce dimensions
Result: Fewer parameters, better performance
```

---

## Convolutional Layer

### Concept
Apply a filter (kernel) across the image to detect patterns.

### How Convolution Works

```
Input Image (5×5):          Filter (3×3):
[1 2 3 4 5]                [0 1 0]
[2 3 4 5 6]                [1 -4 1]
[3 4 5 6 7]                [0 1 0]
[4 5 6 7 8]
[5 6 7 8 9]

Convolution: Slide filter, compute dot product

Position 1:
[1 2 3]         [0 1 0]     = 1×0 + 2×1 + 3×0
[2 3 4]    *    [1 -4 1]      + 2×1 + 3×-4 + 4×1
[3 4 5]         [0 1 0]        + 3×0 + 4×1 + 5×0
              = 2 - 12 + 4 + 4 = -2

Result: (3×3 output for 5×5 input, 3×3 filter)
```

### Parameters

```python
# Kernel size: 3×3
# Stride: 1 (move 1 pixel at a time)
# Padding: 0 (no padding)

Output_Height = (Input_Height - Kernel_Height) / Stride + 1
Output_Width = (Input_Width - Kernel_Width) / Stride + 1
Output_Channels = Number of Filters
```

### Code Example

```python
from tensorflow.keras import layers

# Single Convolutional Layer
conv_layer = layers.Conv2D(
    filters=32,              # Number of filters
    kernel_size=(3, 3),      # Filter size
    strides=(1, 1),          # Step size
    padding='same',          # 'same' or 'valid'
    activation='relu',       # Activation function
    input_shape=(224, 224, 3) # (height, width, channels)
)

# Usage in model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu')
])
```

### Padding Options

```
'valid': No padding (reduces size)
Input: 5×5, Kernel: 3×3, Stride: 1
Output: 3×3

'same': Pad to keep size
Input: 5×5, Kernel: 3×3, Stride: 1
Output: 5×5
```

---

## Pooling Layer

### Purpose
Reduce spatial dimensions while retaining important information.

### Max Pooling
Keep maximum value in each window.

```
Input (4×4):       Max Pool (2×2, stride 2):
[1 2 3 4]         [2 4]
[5 6 7 8]    →    [6 8]
[9 10 11 12]
[13 14 15 16]

Window 1: [1,2,5,6] → max=6 (position 1,1)
Window 2: [3,4,7,8] → max=8 (position 1,2)
etc.
```

### Code Example

```python
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),  # Reduce by 2× in each dimension
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2))
])
```

### Average Pooling
Alternative: Take average instead of max.

```python
layers.AveragePooling2D((2, 2))
```

---

## Complete CNN Architecture

### VGG-like Architecture

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # Block 1
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 3
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## Flattening and Dense Layers

### Transition from Convolutional to Dense

```python
# After convolutions, output shape: (batch, 7, 7, 512)
layers.Flatten(),  # Reshape to (batch, 7×7×512=25,088)

# Then normal dense layers
layers.Dense(512, activation='relu'),
layers.Dense(10, activation='softmax')
```

---

## Common CNN Architectures

### LeNet-5 (1998)
Simple, effective architecture for digit recognition.

### AlexNet (2012)
Breakthrough architecture for ImageNet.

### VGG (2014)
Deep network with small 3×3 filters.

### ResNet (2015)
Skip connections for very deep networks.

```python
from tensorflow.keras.applications import ResNet50

model = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
```

### MobileNet
Lightweight for mobile devices.

```python
from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))
```

---

## Transfer Learning with Pre-trained Models

### Overview
Use weights from model trained on large dataset, fine-tune on your data.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers

# Load pre-trained model (without top classification layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model weights (don't train)
base_model.trainable = False

# Add custom classification layers
model = keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## Image Preprocessing

### Normalization

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Normalize pixel values to [0, 1]
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load images
train_generator = train_datagen.flow_from_directory(
    'train_dir',
    target_size=(224, 224),
    batch_size=32
)

test_generator = test_datagen.flow_from_directory(
    'test_dir',
    target_size=(224, 224),
    batch_size=32
)

# Train
model.fit(train_generator, validation_data=test_generator, epochs=20)
```

---

## Visualizing Filters and Features

### Visualize Learned Filters

```python
import matplotlib.pyplot as plt

# Get weights of first convolutional layer
filters = model.layers[0].get_weights()[0]  # (3, 3, 3, 32)

# filters shape: (height, width, input_channels, num_filters)
# Visualize first 16 filters

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i < filters.shape[3]:
        ax.imshow(filters[:, :, 0, i], cmap='gray')
        ax.set_title(f'Filter {i}')
        ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## Use Cases

### Image Classification
Classify images into categories.

### Object Detection
Locate and classify objects in images.

### Image Segmentation
Label each pixel with class.

### Facial Recognition
Identify faces.

### Medical Imaging
Detect abnormalities in X-rays, CT scans.

---

## Best Practices for CNNs

1. **Start with pre-trained models**: Transfer learning is powerful
2. **Use ImageNet normalization**: For pre-trained models
3. **Data augmentation**: Critical for preventing overfitting
4. **Gradually increase depth**: Don't start with very deep networks
5. **Monitor GPU memory**: CNNs are memory-intensive
6. **Batch normalization**: Helps convergence
7. **Appropriate padding**: Use 'same' to maintain dimensions

---

## References
- LeCun et al. "Gradient-Based Learning Applied to Document Recognition" (1998)
- Krizhevsky et al. "ImageNet Classification with Deep CNNs" (2012)
- Goodfellow et al. "Deep Learning" (MIT Press)

---
