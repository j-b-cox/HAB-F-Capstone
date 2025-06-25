import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_data(X_path="X.npy", y_path="y.npy"):
    """Load feature and label arrays from .npy files."""
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing {X_path} or {y_path}")
    X = np.load(X_path)
    y = np.load(y_path)
    logging.info(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D (n_samples, n_features), got shape {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be 1D with same length as X; got y.shape={y.shape}")
    return X, y

def preprocess(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Reshape for 1D CNN, split into train/val/test, and standardize per wavelength.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    n_samples, n_features = X.shape
    # Reshape for Conv1D: (n_samples, length, 1)
    X_cnn = X.reshape((n_samples, n_features, 1))
    logging.info(f"Reshaped X to CNN input shape: {X_cnn.shape}")

    # train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_cnn, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # train vs val
    # val_size is fraction of trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size, random_state=random_state, stratify=y_trainval
    )
    logging.info(f"Split shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    # Standardize per feature index (wavelength) using training set
    # Flatten to 2D for scaler: (n_samples, n_features)
    def scale_split(X_split, scaler=None):
        n, L, _ = X_split.shape
        flat = X_split.reshape((n, L))
        if scaler is None:
            scaler = StandardScaler()
            flat_scaled = scaler.fit_transform(flat)
        else:
            flat_scaled = scaler.transform(flat)
        # reshape back
        X_scaled = flat_scaled.reshape((n, L, 1))
        return X_scaled, scaler

    X_train_scaled, scaler = scale_split(X_train)
    X_val_scaled, _ = scale_split(X_val, scaler=scaler)
    X_test_scaled, _ = scale_split(X_test, scaler=scaler)
    logging.info("Completed standardization based on training set.")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

def compute_class_weights(y_train):
    """Compute class weights to balance binary classes."""
    classes = np.unique(y_train)
    cw = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, cw))
    logging.info(f"Computed class weights: {class_weights}")
    return class_weights

def build_1d_cnn(input_shape):
    """
    Build a simple 1D CNN for binary classification.
    input_shape: (length, 1)
    """
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test,
                       batch_size=32, epochs=50, model_dir="model_checkpoints"):
    """
    Train the CNN with early stopping and checkpointing, then evaluate on test set.
    """
    input_shape = X_train.shape[1:]  # (length, 1)
    model = build_1d_cnn(input_shape)
    model.summary()

    # Prepare callbacks
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "best_model.h5")
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    mc = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auc', mode='max',
                                   save_best_only=True, verbose=1)

    # Class weights
    class_weights = compute_class_weights(y_train)

    logging.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, mc],
        class_weight=class_weights,
        verbose=2
    )
    logging.info("Training complete. Loading best weights and evaluating on test set.")
    # Load best model weights
    model.load_weights(checkpoint_path)
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=2)
    logging.info(f"Test results - loss: {test_loss:.4f}, accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
    return model, history

def main():
    # Paths can be adjusted
    X_path = "X.npy"
    y_path = "y.npy"

    X, y = load_data(X_path, y_path)
    # Optional: inspect class distribution
    unique, counts = np.unique(y, return_counts=True)
    logging.info(f"Overall class distribution: {dict(zip(unique, counts))}")

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(X, y)
    model, history = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)

    # Save final model
    final_model_path = "spectral_1d_cnn.h5"
    model.save(final_model_path)
    logging.info(f"Saved final model to {final_model_path}")

    # Optionally, save scaler for inference later
    # from joblib import dump
    # dump(scaler, "scaler.joblib")
    # logging.info("Saved scaler to scaler.joblib")

if __name__ == "__main__":
    main()
