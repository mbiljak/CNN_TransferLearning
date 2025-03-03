import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import time
import pandas as pd
import gc

# ---------------------- GPU Configuration ----------------------
# Check if GPU is available and configure memory growth to avoid memory allocation errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("\nGPU detected:")
    for device in physical_devices:
        print(f"- {device}")
    try:
         # Enable memory growth to avoid allocating all GPU memory at once
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled")
    except Exception as e:
        print(f"Memory growth error: {e}")
else:
    print("\nGPU not detected - using CPU")

# Reduce TensorFlow verbosity
tf.get_logger().setLevel('ERROR')
tf.debugging.set_log_device_placement(False)

# ----------------------------------------------------------------

# Start timing the execution
start_time = time.time()

# Load CIFAR-10 dataset (32x32 color images in 10 classes)
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Reduced dataset size to save memory and computation time
TRAIN_SAMPLES = 45000 #50000
TEST_SAMPLES = 8000 #10000

# Randomly select subset of samples
indices_train = np.random.choice(len(x_train), TRAIN_SAMPLES, replace=False)
indices_test = np.random.choice(len(x_test), TEST_SAMPLES, replace=False)
x_train = x_train[indices_train]
y_train = y_train[indices_train]
x_test = x_test[indices_test]
y_test = y_test[indices_test]

print(f"Using {len(x_train)} training samples and {len(x_test)} test samples")

# Normalize pixel values to range [0,1]
x_train_norm = x_train.astype('float32') 
x_test_norm = x_test.astype('float32') 

# Convert class vectors to one-hot encoded matrices
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# Define data augmentation to improve model generalization
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"), # Randomly flip images horizontally
    layers.RandomRotation(0.1), # Randomly rotate images by up to 10%
])

# Define a custom CNN architecture for CIFAR-10 classification
def create_efficient_cnn():
    """Create a custom CNN model with increasing filter sizes and batch normalization"""
    model = models.Sequential([
        # Input layer specifying image dimensions
        layers.InputLayer(input_shape=(32, 32, 3)),
        
        # First convolutional block
        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),    # Normalize activations for faster training
        layers.MaxPooling2D((2, 2)),    # Reduce spatial dimensions by half
        
        # Second convolutional block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        # Fourth convolutional block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        # Fifth convolutional block
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),               # Convert 3D feature maps to 1D feature vectors
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),            # Prevent overfitting by randomly dropping 50% of neurons
        layers.Dense(10, activation='softmax')  # Output layer with 10 classes
    ])
    return model

# Create and compile the custom CNN model
model_scratch = create_efficient_cnn()
model_scratch.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),  # Learning rate 0.001
    loss='categorical_crossentropy',            # Standard loss for multi-class classification
    metrics=['accuracy']                        # Monitor accuracy during training
)

# Define callbacks for training optimization
callbacks = [
    # Stop training when validation accuracy stops improving
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10),
    # Reduce learning rate when validation loss plateaus
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

BATCH_SIZE = 64 # Number of samples per gradient update

# Train the custom CNN model
print("\nTraining custom CNN...")
history_scratch = model_scratch.fit(
    x_train_norm, y_train_cat,
    epochs=15,                     # Maximum number of training epochs
    batch_size=BATCH_SIZE,
    validation_data=(x_test_norm, y_test_cat),
    callbacks=callbacks,
    verbose=1
)

# Create a feature extractor from the trained CNN for use with SVM
def create_feature_extractor():
    """Create feature extractor from trained CNN"""
    # Create a new input layer with the same shape as the original model
    input_shape = (32, 32, 3)  # CIFAR-10 image dimensions
    inputs = tf.keras.Input(shape=input_shape)
    
    # Create a new model that processes inputs through the first 8 layers of model_scratch
    x = inputs
    # Apply the first 8 layers (up to and including the last Conv2D layer)
    # Sequential model layers are zero-indexed, so we use layers 0-7
    for i in range(8):
        x = model_scratch.layers[i](x)
    
    # Create and return the feature extractor model
    return tf.keras.Model(inputs=inputs, outputs=x)

# Create the feature extractor
feature_extractor = create_feature_extractor()
print("Feature extractor summary:")
feature_extractor.summary()

# Extract features in batches (keep the rest of the code the same)
def extract_features_in_batches(model, data, batch_size=64):
    features_list = []
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        batch = data[i:i+batch_size]
        features = model.predict(batch, verbose=0)
        features_list.append(features)
    # Concatenate all batches and flatten the spatial dimensions
    return np.concatenate(features_list, axis=0).reshape(num_samples, -1)

# Extract CNN features for use with SVM
print("\nExtracting CNN features for SVM...")
x_train_features = extract_features_in_batches(feature_extractor, x_train_norm)
x_test_features = extract_features_in_batches(feature_extractor, x_test_norm)

# Apply PCA to reduce feature dimensionality for SVM
print("Applying PCA...")
pca = PCA(n_components=100)
x_train_pca = pca.fit_transform(x_train_features)
x_test_pca = pca.transform(x_test_features)

# Train SVM
print("Training SVM...")
svm = SVC(kernel='rbf', C=1, gamma='scale') # Radial basis function kernel
svm.fit(x_train_pca, y_train.flatten())

# Evaluate SVM
svm_accuracy = accuracy_score(y_test.flatten(), svm.predict(x_test_pca))
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# Free memory by deleting large feature arrays
del x_train_features, x_test_features, x_train_pca, x_test_pca
gc.collect()

# ---------------------- Transfer Learning with ResNet50 ----------------------
# Set up transfer learning using pre-trained ResNet50
print("\nSetting up transfer learning with ResNet50...")

# Load pre-trained ResNet50 without the classification head
base_model = applications.ResNet50(
    weights='imagenet',     # Pre-trained on ImageNet
    include_top=False,      # Exclude classification layers
    input_shape=(128, 128, 3)  # ResNet50 expected input size
)
base_model.trainable = False # Freeze weights initially

# Build a model on top of ResNet50
inputs = layers.Input(shape=(32, 32, 3)) # CIFAR-10 image size
x = data_augmentation(inputs) # Apply data augmentation

# Resize CIFAR-10 images to fit ResNet50 input (using 128x128 instead of 224x224 for efficiency)
x = layers.Lambda(lambda img: tf.image.resize(img, (128, 128)))(x) 

# Preprocess for ResNet50
x = applications.resnet50.preprocess_input(x)
# Extract features using ResNet50
x = base_model(x)
# Add classification head
x = layers.GlobalAveragePooling2D()(x)   # Pool spatial dimensions
x = layers.Dense(128, activation='relu')(x)  # Bottleneck layer
outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

# Create the transfer learning model
model_transfer = tf.keras.Model(inputs, outputs)

# Compile the model
model_transfer.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# First phase: Train only the top layers (base model frozen)
print("Training transfer learning model (frozen)...")
history_transfer_frozen = model_transfer.fit(
    x_train_norm, y_train_cat,
    epochs=5,
    batch_size=BATCH_SIZE,
    validation_data=(x_test_norm, y_test_cat),
    callbacks=callbacks,
    verbose=1
)

# Second phase: Fine-tune some of the ResNet50 layers
# Unfreeze the base model
base_model.trainable = True
# Freeze early layers and only train deeper layers
for layer in base_model.layers[:-30]: # Keep the last 30 layers trainable
    layer.trainable = False

# Recompile with lower learning rate to avoid destroying pre-trained weights
model_transfer.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001), # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train with fine-tuning
print("Fine-tuning transfer learning model...")
history_transfer = model_transfer.fit(
    x_train_norm, y_train_cat,
    epochs=5,
    batch_size=BATCH_SIZE,
    validation_data=(x_test_norm, y_test_cat),
    callbacks=callbacks,
    verbose=1
)

# Combine training histories from frozen and fine-tuning phases
def combine_histories(h1, h2):
    return {
        'accuracy': h1.history['accuracy'] + h2.history['accuracy'],
        'val_accuracy': h1.history['val_accuracy'] + h2.history['val_accuracy'],
        'loss': h1.history['loss'] + h2.history['loss'],
        'val_loss': h1.history['val_loss'] + h2.history['val_loss']
    }

combined_history_transfer = combine_histories(history_transfer_frozen, history_transfer)

# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"\nTotal runtime: {elapsed_time/60:.2f} minutes")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_scratch.history['accuracy'], label='CNN Training')
plt.plot(history_scratch.history['val_accuracy'], label='CNN Testing')
plt.plot(combined_history_transfer['accuracy'], label='ResNet50 Training')
plt.plot(combined_history_transfer['val_accuracy'], label='ResNet50 Training')
plt.title('Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_scratch.history['loss'], label='CNN Training')
plt.plot(history_scratch.history['val_loss'], label='CNN Testing')
plt.plot(combined_history_transfer['loss'], label='ResNet50 Training')
plt.plot(combined_history_transfer['val_loss'], label='ResNet50 Testing')
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_comparison.png')
plt.show()

# Performance summary
summary_data = {
    'Model': ['Custom CNN', 'ResNet50 Transfer', 'CNN+SVM'],
    'Accuracy': [
        history_scratch.history['val_accuracy'][-1],
        combined_history_transfer['val_accuracy'][-1],
        svm_accuracy
    ]
}
summary_df = pd.DataFrame(summary_data)
print("\nPerformance Summary:")
print(summary_df.to_markdown(index=False))
