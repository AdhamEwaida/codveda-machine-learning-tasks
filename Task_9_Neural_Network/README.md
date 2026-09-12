# Level 3 - Task 3: Neural Network with TensorFlow/Keras

This task builds a feed-forward neural network for handwritten-digit classification using TensorFlow/Keras and the scikit-learn Digits dataset.

## What the script does

- Loads 1,797 labeled 8x8 grayscale digit images from classes 0 through 9.
- Normalizes pixel intensities from the range 0-16 to 0-1.
- Creates stratified training, validation, and held-out test sets.
- Builds a Keras network with 128-unit and 64-unit hidden layers.
- Uses ReLU activations, dropout regularization, and a 10-class softmax output.
- Trains with backpropagation and Adam optimization.
- Uses early stopping and restores the best validation-loss weights.
- Evaluates accuracy plus macro precision, recall, and F1-score.
- Saves the trained `.keras` model, predictions, reports, and plots.

The dataset is bundled with scikit-learn, so the task does not need an internet download at runtime.

## Run

From the repository root:

```powershell
python .\Task_9_Neural_Network\task9_neural_network.py
```

## Generated files

- `digits_data.csv` - flattened image pixels and digit labels.
- `outputs/digits_neural_network.keras` - trained Keras model.
- `outputs/model_architecture.txt` - layer shapes and parameter counts.
- `outputs/training_history.csv` - per-epoch loss and accuracy.
- `outputs/evaluation_metrics.csv` - held-out test metrics.
- `outputs/classification_report.txt` - per-digit classification metrics.
- `outputs/test_predictions.csv` - actual labels, predictions, confidence, and class probabilities.
- `outputs/training_curves.png` - training/validation loss and accuracy.
- `outputs/confusion_matrix.png` - held-out test confusion matrix.
- `outputs/sample_predictions.png` - example images with predictions.
- `outputs/summary.txt` - architecture, training, and evaluation summary.
