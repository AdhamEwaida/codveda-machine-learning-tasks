"""Level 3 - Task 3: feed-forward neural network with TensorFlow/Keras."""

import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import tensorflow as tf


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def normalize_keras_archive(model_path: Path) -> None:
    """Remove save-time metadata so repeated deterministic runs create one file."""
    def normalize_shared_ids(value: object) -> None:
        if isinstance(value, dict):
            if "shared_object_id" in value:
                value["shared_object_id"] = 1
            for child in value.values():
                normalize_shared_ids(child)
        elif isinstance(value, list):
            for child in value:
                normalize_shared_ids(child)

    temporary_path = model_path.with_suffix(".normalized.keras")
    fixed_timestamp = (2020, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(model_path, "r") as source, zipfile.ZipFile(
        temporary_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as destination:
        for source_info in source.infolist():
            content = source.read(source_info.filename)
            if source_info.filename == "metadata.json":
                metadata = json.loads(content.decode("utf-8"))
                metadata["date_saved"] = "2020-01-01@00:00:00"
                content = json.dumps(metadata, sort_keys=True).encode("utf-8")
            elif source_info.filename == "config.json":
                configuration = json.loads(content.decode("utf-8"))
                normalize_shared_ids(configuration)
                content = json.dumps(configuration, sort_keys=True).encode("utf-8")
            target_info = zipfile.ZipInfo(source_info.filename, fixed_timestamp)
            target_info.compress_type = zipfile.ZIP_DEFLATED
            target_info.external_attr = source_info.external_attr
            destination.writestr(target_info, content)
    temporary_path.replace(model_path)


def build_model(number_of_features: int, number_of_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(number_of_features,), name="pixels"),
            tf.keras.layers.Dense(128, activation="relu", name="hidden_128"),
            tf.keras.layers.Dropout(0.20, name="dropout"),
            tf.keras.layers.Dense(64, activation="relu", name="hidden_64"),
            tf.keras.layers.Dense(
                number_of_classes, activation="softmax", name="digit_probabilities"
            ),
        ],
        name="digits_feed_forward_network",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    digits = load_digits()
    X = digits.data.astype("float32")
    y = digits.target.astype("int64")
    feature_names = [f"pixel_{index}" for index in range(X.shape[1])]
    exported_data = pd.DataFrame(X, columns=feature_names)
    exported_data["digit"] = y
    exported_data.to_csv(BASE_DIR / "digits_data.csv", index=False)

    # Pixel values in this dataset range from 0 to 16.
    X = X / 16.0
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_validation,
    )

    model = build_model(X.shape[1], len(digits.target_names))
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=0,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
        shuffle=True,
    )

    test_loss, keras_accuracy = model.evaluate(X_test, y_test, verbose=0)
    probabilities = model.predict(X_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "test_loss": test_loss,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "keras_accuracy_check": keras_accuracy,
    }
    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "evaluation_metrics.csv", index=False
    )

    history_table = pd.DataFrame(history.history)
    history_table.insert(0, "epoch", np.arange(1, len(history_table) + 1))
    history_table.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    prediction_table = pd.DataFrame(
        {
            "actual_digit": y_test,
            "predicted_digit": predictions,
            "correct": predictions == y_test,
            "confidence": probabilities.max(axis=1),
        }
    )
    for digit in digits.target_names:
        prediction_table[f"probability_{digit}"] = probabilities[:, digit]
    prediction_table.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    report = classification_report(
        y_test,
        predictions,
        labels=digits.target_names,
        target_names=[str(digit) for digit in digits.target_names],
        digits=4,
        zero_division=0,
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    architecture_lines: list[str] = []
    model.summary(print_fn=architecture_lines.append)
    (OUTPUT_DIR / "model_architecture.txt").write_text(
        "\n".join(architecture_lines), encoding="utf-8"
    )
    model_path = OUTPUT_DIR / "digits_neural_network.keras"
    model.save(model_path)
    normalize_keras_archive(model_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history_table["epoch"], history_table["loss"], label="Training")
    axes[0].plot(
        history_table["epoch"], history_table["val_loss"], label="Validation"
    )
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        history_table["epoch"], history_table["accuracy"], label="Training"
    )
    axes[1].plot(
        history_table["epoch"],
        history_table["val_accuracy"],
        label="Validation",
    )
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close(fig)

    matrix = confusion_matrix(y_test, predictions, labels=digits.target_names)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=digits.target_names,
        yticklabels=digits.target_names,
    )
    plt.xlabel("Predicted digit")
    plt.ylabel("Actual digit")
    plt.title("Neural Network Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    sample_indices = np.linspace(0, len(X_test) - 1, 12, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for axis, sample_index in zip(axes.ravel(), sample_indices):
        axis.imshow(X_test[sample_index].reshape(8, 8), cmap="gray_r")
        actual = y_test[sample_index]
        predicted = predictions[sample_index]
        color = "green" if actual == predicted else "red"
        axis.set_title(f"Actual: {actual} | Pred: {predicted}", color=color)
        axis.axis("off")
    fig.suptitle("Sample Digit Predictions")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=160)
    plt.close(fig)

    best_epoch = int(history_table["val_loss"].idxmin() + 1)
    summary = f"""LEVEL 3 - TASK 3: NEURAL NETWORK WITH TENSORFLOW/KERAS

Dataset: scikit-learn Digits dataset
Records: {len(X)}
Input features: {X.shape[1]} normalized pixel values
Classes: 10 digits (0-9)
Split: {len(X_train)} training / {len(X_validation)} validation / {len(X_test)} testing
Architecture: Input(64) -> Dense(128, ReLU) -> Dropout(0.20) -> Dense(64, ReLU) -> Dense(10, Softmax)
Optimizer: Adam (learning_rate=0.001)
Loss: sparse categorical cross-entropy
Training: backpropagation with a maximum of 50 epochs and early stopping
Epochs completed: {len(history_table)}
Best validation-loss epoch: {best_epoch}
Trainable parameters: {model.count_params()}

Held-out test metrics
- Loss: {metrics['test_loss']:.4f}
- Accuracy: {metrics['accuracy']:.4f}
- Macro precision: {metrics['precision_macro']:.4f}
- Macro recall: {metrics['recall_macro']:.4f}
- Macro F1-score: {metrics['f1_macro']:.4f}
"""
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
