import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_dir, val_dir, epochs=EPOCHS):
    # ✅ Data augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    model = build_model()
    model.fit(train_data, validation_data=val_data, epochs=epochs)

    return model

def evaluate_model(model, test_dir):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_data = test_datagen.flow_from_directory(test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='binary')

    loss, acc = model.evaluate(test_data)
    print(f"\n✅ Test Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")

def load_model(weights_path="weights/model_weights.weights.h5"):
    model = build_model()
    model.load_weights(weights_path)
    return model

def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)[0][0]
    label = "Accident" if pred >= 0.5 else "No Accident"
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, confidence

if __name__ == "__main__":
    train_dir = "data/train"
    val_dir = "data/val"
    test_dir = "data/test"

    os.makedirs("weights", exist_ok=True)

    print("\n🔧 Training model...")
    model = train_model(train_dir, val_dir, epochs=EPOCHS)

    print("\n💾 Saving weights...")
    model.save_weights("weights/model_weights.weights.h5")

    print("\n📊 Evaluating on test set...")
    evaluate_model(model, test_dir)
