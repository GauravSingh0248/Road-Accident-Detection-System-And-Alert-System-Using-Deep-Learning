import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 12
FINE_TUNE_AT = 100


def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model, base_model


def build_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "weights/best_model.weights.h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]


def make_data_generators(train_dir, val_dir):
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
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
    )

    # Keep deterministic ordering for threshold tuning.
    val_data = eval_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
    )
    return train_data, val_data


def train_model(train_dir, val_dir, head_epochs=HEAD_EPOCHS, fine_tune_epochs=FINE_TUNE_EPOCHS):
    train_data, val_data = make_data_generators(train_dir, val_dir)
    model, base_model = build_model()

    print("\nStage 1: Train classifier head")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=head_epochs,
        callbacks=build_callbacks(),
    )

    print("\nStage 2: Fine-tune top MobileNetV2 layers")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=fine_tune_epochs,
        callbacks=build_callbacks(),
    )

    return model, val_data


def _binary_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def tune_threshold(model, val_data):
    val_data.reset()
    probs = model.predict(val_data, verbose=0).ravel()
    y_true = val_data.classes.astype(int)

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.30, 0.71, 0.01):
        preds = (probs >= threshold).astype(int)
        metrics = _binary_metrics(y_true, preds)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(round(threshold, 2))

    return best_threshold, best_f1


def evaluate_model(model, test_dir, threshold=0.5):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
    )

    loss, acc = model.evaluate(test_data, verbose=0)
    test_data.reset()
    probs = model.predict(test_data, verbose=0).ravel()
    y_true = test_data.classes.astype(int)
    y_pred = (probs >= threshold).astype(int)
    metrics = _binary_metrics(y_true, y_pred)

    print(f"\nTest Accuracy (Keras): {acc*100:.2f}% | Loss: {loss:.4f}")
    print(f"Threshold used: {threshold:.2f}")
    print("Confusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(f"[[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]")
    print(
        "Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}".format(
            metrics["precision"], metrics["recall"], metrics["f1"]
        )
    )

def load_model(weights_path="weights/model_weights.weights.h5"):
    model, _ = build_model()
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

    print("\nTraining model...")
    model, val_data = train_model(train_dir, val_dir)

    print("\nTuning threshold on validation set...")
    best_threshold, best_f1 = tune_threshold(model, val_data)
    print(f"Best threshold: {best_threshold:.2f} | Validation F1: {best_f1:.4f}")

    print("\nSaving weights...")
    model.save_weights("weights/model_weights.weights.h5")

    print("\nEvaluating on test set...")
    evaluate_model(model, test_dir, threshold=best_threshold)
