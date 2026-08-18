# 📚 Deep Learning Glossary & Cheat Sheet

A comprehensive reference guide for all concepts used in the Heart Disease Prediction ANN project.

---

## 📖 Table of Contents

1. [Machine Learning Basics](#machine-learning-basics)
2. [Neural Network Concepts](#neural-network-concepts)
3. [Training Concepts](#training-concepts)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Data Preprocessing](#data-preprocessing)
6. [Common Algorithms](#common-algorithms)
7. [Error Messages & Solutions](#error-messages--solutions)
8. [Code Snippets Cheat Sheet](#code-snippets-cheat-sheet)
9. [Quick Reference Tables](#quick-reference-tables)

---

## Machine Learning Basics

### **Artificial Intelligence (AI)**
- **Definition**: Computer systems designed to perform tasks that typically require human intelligence
- **Domains**: Robotics, NLP, Computer Vision, Prediction, etc.
- **Our Project**: Predicting disease is an AI task ✓

### **Machine Learning (ML)**
- **Definition**: Subset of AI where systems learn from data without being explicitly programmed
- **Process**: Data → Algorithm → Model → Predictions
- **Our Project**: Model learns patterns in patient data ✓

### **Deep Learning (DL)**
- **Definition**: Subset of ML using neural networks with multiple layers
- **Advantage**: Can learn very complex patterns automatically
- **Our Project**: Using deep neural networks (ANN) ✓

### **Supervised Learning**
- **Definition**: Learning from labeled data (input + correct output)
- **Example**: Training on patients with known diagnoses
- **Our Project**: We know who has/doesn't have disease ✓

### **Classification**
- **Definition**: Predicting which category something belongs to
- **Types**: Binary (2 classes), Multiclass (>2 classes)
- **Our Project**: Binary classification (disease or no disease) ✓

### **Regression**
- **Definition**: Predicting continuous numerical values
- **Example**: Predicting house price or temperature
- **Our Project**: Not used (we're doing classification) ✗

---

## Neural Network Concepts

### **Artificial Neural Network (ANN)**
```
Definition: Computing system inspired by biological neurons

Structure:
┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐
│ Input Layer │→→│ Hidden Layer │→→│ Hidden Layer │→→│Output Layer │
│  (13 nodes) │  │  (16 nodes)  │  │  (8 nodes)   │  │ (1 node)    │
└─────────────┘  └──────────────┘  └──────────────┘  └─────────────┘

Purpose: Learn complex patterns from data
Our project: Diagnose heart disease ✓
```

### **Neuron (Perceptron)**
```
Definition: Basic computational unit in neural network

Process:
1. Receive inputs: x₁, x₂, x₃, ...
2. Multiply by weights: w₁, w₂, w₃, ...
3. Add bias: b
4. Apply activation: f(sum)
5. Output: y

Formula: y = f(Σ(wᵢ × xᵢ) + b)

Example:
Input: age=45, sex=1, chol=250
Weights: w₁=0.5, w₂=0.3, w₃=0.2
Bias: b=0.1
Calculation:
  sum = (0.5×45) + (0.3×1) + (0.2×250) + 0.1
      = 22.5 + 0.3 + 50 + 0.1
      = 72.9
  y = activation(72.9)
```

### **Layer**
```
Definition: Group of neurons working together

Types:
1. Input Layer
   └─ One neuron per feature
   └─ Our project: 13 neurons (13 features)

2. Hidden Layers
   └─ Learn patterns from previous layer
   └─ Our project: 2 hidden layers (16 and 8 neurons)

3. Output Layer
   └─ Produces final prediction
   └─ Our project: 1 neuron (disease yes/no)
```

### **Weight (w)**
```
Definition: Parameter that scales/amplifies input importance

Range: -∞ to +∞
Initial: Random values
Updated: During training via backpropagation
Impact: Large weight = Feature strongly influences output
        Small weight = Feature weakly influences output

Example:
If age has weight=0.8 and sex has weight=0.1
→ Age is 8x more important than sex
```

### **Bias (b)**
```
Definition: Constant added to sum before activation

Purpose: Shift the activation function
Range: -∞ to +∞
Analogy: Like threshold in decision-making

Example:
Without bias: y = f(Σ(w×x))
With bias:    y = f(Σ(w×x) + b)
The bias allows model to learn offset patterns
```

### **Activation Function**
```
Definition: Non-linear function applied to neuron output
Purpose: Enable network to learn complex patterns
Without it: Network becomes just linear regression

Key Activations:

1. ReLU (Rectified Linear Unit)
   Formula: f(x) = max(0, x)
   Range: 0 to +∞
   Use: Hidden layers
   Pros: Simple, fast, prevents vanishing gradient
   Cons: Can have "dead neurons"
   
   Graph:
   y
   │      ╱
   │     ╱
   └────╱────── x
        0
   
2. Sigmoid
   Formula: σ(x) = 1 / (1 + e^(-x))
   Range: 0 to 1 (Perfect for probability!)
   Use: Output layer (binary classification)
   Pros: Output = probability
   Cons: Vanishing gradient for extreme values
   
   Graph:
   y
   1│        ╱────
    │       ╱
   0.5┤─────╱──
    │   ╱
   0│──╱────── x
   
3. Tanh (Hyperbolic Tangent)
   Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   Range: -1 to 1
   Use: Hidden layers, sometimes output
   Pros: Better than sigmoid
   Cons: Still vanishing gradient
   
4. Softmax
   Use: Multiclass classification (>2 classes)
   Output: Probability distribution (sum=1)
```

### **Dropout**
```
Definition: Regularization technique preventing overfitting

Process:
- Randomly disable 20% of neurons during training
- Each epoch: Different random neurons disabled
- During prediction: All neurons active

Analogy: Like studying with a team:
- Sometimes one person is absent (simulating pressure)
- Remaining members learn to work independently
- Final exam: Everyone participates

Benefits:
✓ Prevents overfitting
✓ Reduces co-adaptation
✓ Improves generalization

Our project: 20% dropout after each hidden layer
```

### **Dense Layer (Fully Connected)**
```
Definition: Every neuron connects to every neuron in next layer

Visualization:
Layer 1        Layer 2
 ●──────────●
 ├──────────┤
 ●─┼────────●
 ├─┼────────┤
 ●─┼────────●
   └────────●

Calculation:
For each neuron in next layer:
output = activation(sum(all_inputs × weights) + bias)

Full connectivity = Learns all relationships
```

---

## Training Concepts

### **Epoch**
```
Definition: One complete pass through entire training dataset

Process:
Epoch 1: Train on all 242 training samples → Loss calculated
Epoch 2: Train again on same 242 samples → Loss recalculated
...
Epoch 100: Final pass through data

Typical: 50-200 epochs
Too few: Model hasn't learned enough
Too many: Wastes time, risk of overfitting
```

### **Batch**
```
Definition: Subset of training data processed together

Example:
Total training data: 242 samples
Batch size: 32
Batches per epoch: 242 / 32 = ~8 batches

Process:
Batch 1: Sample 1-32 → Update weights
Batch 2: Sample 33-64 → Update weights
...
Batch 8: Sample 225-242 → Epoch complete

Why batches?
✓ Faster: Update weights multiple times per epoch
✓ Stability: Gradient is more stable than single sample
✓ Memory: Don't need all data in memory
```

### **Gradient Descent**
```
Definition: Algorithm to minimize loss by updating weights

Process:
1. Calculate loss
2. Calculate gradient (direction of steepest increase)
3. Move in opposite direction to decrease loss
4. Repeat

Visualization:
        Loss
        ╱│╲
       ╱ │ ╲
      ╱  │  ╲    ← Gradient (slope)
     ╱   │   ╲
    ╱    │    ╲
   ╱ ←←← ┃ ←←← ╲  (Move down)
  ╱      ┃      ╲
 ╱       ┃       ╲
────────┃────────────  Weights
        Minimum

Learning rate: How big steps to take
- Too small: Slow convergence
- Too large: Might overshoot minimum
```

### **Learning Rate**
```
Definition: Controls step size in gradient descent

Range: Typically 0.00001 to 0.1
Our project: 0.001

Impact:
┌──────────────────────────────────────────┐
│ Learning Rate = 0.0001 (Too small)       │
│ Loss: Decreases very slowly              │
│ Time: Takes forever to converge          │
│ Issue: Gets stuck in local minima        │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ Learning Rate = 0.001 (Good)             │
│ Loss: Decreases smoothly                 │
│ Time: Reasonable convergence             │
│ Result: Finds good minimum               │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ Learning Rate = 0.1 (Too large)          │
│ Loss: Bounces around, doesn't improve    │
│ Time: Diverges (gets worse!)             │
│ Issue: Overshoots minimum                │
└──────────────────────────────────────────┘
```

### **Backpropagation**
```
Definition: Algorithm to calculate gradients for weight updates

Process:
1. Forward Pass: x → Network → Prediction
2. Calculate Loss: Prediction vs Actual
3. Backward Pass: Calculate gradient for each weight
4. Update Weights: w = w - (learning_rate × gradient)

Mathematical:
∂L/∂w = gradient of loss with respect to weight
w_new = w_old - α × ∂L/∂w
(where α is learning rate)

Chain Rule: Multiply gradients from each layer back
```

### **Optimizer**
```
Definition: Algorithm for updating weights during training

Common Optimizers:

1. SGD (Stochastic Gradient Descent)
   Simple but effective
   Sometimes gets stuck locally

2. Adam (Adaptive Moment Estimation) - OUR PROJECT
   ✓ Adapts learning rate per parameter
   ✓ Usually fastest convergence
   ✓ Best for most problems
   ✓ Our project uses this

3. RMSprop
   Good middle ground
   Less popular than Adam

4. Adagrad
   Adapts learning rate
   Good for sparse data
```

### **Loss Function**
```
Definition: Measures how wrong predictions are

Formula: loss = measure_of_error(predictions, actual)

Lower loss = Better predictions
Our project uses: Binary Crossentropy

Binary Crossentropy:
Purpose: For binary classification (0/1)
Formula: -[y×log(ŷ) + (1-y)×log(1-ŷ)]

Example:
Actual = 1 (has disease)
Prediction = 0.9 (90% confident has disease)
Loss = -[1×log(0.9) + 0×log(0.1)] = 0.105 (Low error ✓)

Actual = 1 (has disease)
Prediction = 0.2 (only 20% confident has disease)
Loss = -[1×log(0.2) + 0×log(0.8)] = 1.609 (High error ✗)
```

### **Overfitting**
```
Definition: Model memorizes training data instead of learning patterns

Visual:
Training Accuracy: 99% ✓
Testing Accuracy: 60% ✗

Signs:
- Training loss decreases but validation loss increases
- Large gap between training and validation accuracy
- Model performs well on training, poor on new data

Solutions:
1. Use Dropout (we do! ✓)
2. Reduce model complexity
3. Add more training data
4. Use early stopping (we do! ✓)
5. Increase regularization
```

### **Underfitting**
```
Definition: Model too simple to capture patterns

Visual:
Training Accuracy: 60% ✗
Testing Accuracy: 58% ✗

Signs:
- Both training and testing accuracy are low
- Validation accuracy plateaus early
- Model is too simple for the data

Solutions:
1. Make model more complex (more neurons/layers)
2. Train longer (more epochs)
3. Increase learning rate
4. Reduce dropout
5. Reduce regularization
```

### **Early Stopping**
```
Definition: Stop training when validation performance plateaus

Process:
- Monitor validation loss after each epoch
- If no improvement for N epochs (patience), stop
- Restore weights from best epoch

Our project: patience=15 epochs

Benefits:
✓ Saves time
✓ Prevents overfitting
✓ Finds optimal number of epochs

Analogy: Stop studying when you're not learning anymore
```

---

## Evaluation Metrics

### **Accuracy**
```
Definition: Percentage of correct predictions

Formula: accuracy = (TP + TN) / (TP + TN + FP + FN)

Example:
Made 100 predictions
90 correct, 10 wrong
Accuracy = 90% ✓

Pros: Simple, interpretable
Cons: Misleading with imbalanced classes

When to use: Balanced datasets
When NOT to use: When class imbalance is severe
```

### **Precision**
```
Definition: Of predicted positive cases, how many were correct?

Formula: precision = TP / (TP + FP)

Example:
Predicted 20 patients have disease
Actually 15 have disease, 5 don't
Precision = 15/20 = 75%

Interpretation:
- 75% of "disease" predictions are correct
- 25% are false alarms

When to use: Want to minimize false positives
Medical: Want to avoid unnecessary treatments
Spam email: Want to avoid blocking real emails
```

### **Recall (Sensitivity)**
```
Definition: Of actual positive cases, how many did we find?

Formula: recall = TP / (TP + FN)

Example:
Actually 20 patients have disease
Found 15 of them
Recall = 15/20 = 75%

Interpretation:
- Caught 75% of disease cases
- Missed 25% (dangerous!)

When to use: Want to minimize false negatives
Medical: Want to catch all disease cases
Fraud detection: Must catch fraudulent transactions
Cancer screening: Must identify all cancers
```

### **F1-Score**
```
Definition: Harmonic mean of precision and recall

Formula: F1 = 2 × (precision × recall) / (precision + recall)

Range: 0 to 1 (higher is better)

When to use: When you care about both precision and recall
Imbalanced classes: Good metric for imbalanced data
Our project: Use this for main evaluation ✓

Example:
Precision = 0.80, Recall = 0.90
F1 = 2 × (0.80 × 0.90) / (0.80 + 0.90)
   = 1.44 / 1.70
   = 0.847 ✓
```

### **Confusion Matrix**
```
Definition: Table showing prediction correctness

Layout:
                    PREDICTED
                 No Disease   Disease
    ┌──────────────────────────────────┐
A   │ TN (True Neg)   FP (False Pos)   │
C   │ Correct no-dis  Wrong: no→disease│
T   │                                  │
U   ├──────────────────────────────────┤
A   │ FN (False Neg)  TP (True Pos)    │
L   │ Wrong: dis→no   Correct disease  │
    └──────────────────────────────────┘

Example:
                Predicted
              No Disease  Disease
Actual  No        42         3
        Yes        7        13

Interpretation:
- TP (13): Correctly identified disease ✓
- TN (42): Correctly identified no disease ✓
- FP (3): False alarm - treated healthy as disease ✗
- FN (7): Missed disease - treated disease as healthy ✗✗ Dangerous!

Calculations:
- Accuracy = (42+13)/65 = 84.6%
- Precision = 13/(3+13) = 81.3%
- Recall = 13/(13+7) = 65%
```

### **ROC-AUC Score**
```
Definition: Receiver Operating Characteristic Area Under Curve

Purpose: Evaluate classification performance across all thresholds

Range: 0.5 to 1.0
- 0.5 = Random guessing (bad)
- 0.7-0.8 = Acceptable
- 0.8-0.9 = Excellent ← Our goal
- 0.9+ = Outstanding

ROC Curve:
Plots: True Positive Rate vs False Positive Rate

Visual:
           TPR
          1 │     ╱─────  (Good model)
    ┌──────┼────╱─────┐
    │      │  ╱ ROC   │
  0.5│──────┼─╱────── │ (Random)
    │      │╱         │
    │      │/          │
          0└────────── FPR
              1

Interpretation:
- Higher curve = Better model
- Curve closer to top-left = Better
- Area under curve = AUC score
```

---

## Data Preprocessing

### **Standardization (Normalization)**
```
Definition: Scale features to have mean=0, std=1

Why needed?
- Features have different scales (age: 0-100, chol: 100-600)
- Neural networks learn faster with normalized data
- Gradient descent converges better

Formula:
x_scaled = (x - mean) / standard_deviation

Example:
Feature: Cholesterol
Original: [126, 250, 500, 564]
Mean: 360, Std: 186

Scaled:
- 126 → (126-360)/186 = -1.26
- 250 → (250-360)/186 = -0.59
- 500 → (500-360)/186 = 0.75
- 564 → (564-360)/186 = 1.10

Result: All features now in range [-3, +3]
```

### **Train-Test Split**
```
Definition: Divide data into training and testing sets

Ratio: Typically 80-20 or 70-30
Our project: 80-20 split

Process:
Total data: 303 samples
├─ Training (80%): 242 samples
│  └─ Used to teach the model
└─ Testing (20%): 61 samples
   └─ Evaluate on unseen data

Why separate?
✓ Prevents data leakage
✓ True estimate of model performance
✓ Detects overfitting

Important: Never train and test on same data!
```

### **Feature Engineering**
```
Definition: Create new features from existing ones

Purpose: Help model learn better patterns

Techniques:
1. Polynomial features: x² for non-linear relationships
2. Interaction features: x₁ × x₂
3. Binning: Convert continuous to categories
4. Domain knowledge: Create medically meaningful features

Example from our project:
Original features: age, sex, cholesterol, ...
Engineered: age_groups (young, middle, old)
           cholesterol_risk (low, normal, high)
```

### **Handling Missing Values**
```
Definition: Deal with null/NaN values in data

Methods:
1. Drop: Remove rows with missing values
   Pros: Simple
   Cons: Lose data

2. Fill with mean: Replace NaN with average
   Pros: Keep all data
   Cons: Loses information

3. Fill with median: Better for skewed data
   Pros: Robust to outliers
   Cons: Less accurate than other methods

4. Forward fill: Use previous value (time series)

5. Interpolate: Estimate based on neighbors

Our project: No missing values ✓
Dataset is clean
```

### **Handling Outliers**
```
Definition: Deal with extreme/unusual values

Detection:
- Z-score: |z| > 3 suggests outlier
- IQR method: Values beyond Q1-1.5×IQR or Q3+1.5×IQR

Handling:
1. Remove: Delete outlier rows
2. Cap: Set to min/max threshold
3. Transform: Log transform to reduce impact
4. Keep: If valid, don't remove

Our project: 
No obvious outliers detected ✓
Medical measurements naturally vary
```

---

## Common Algorithms

### **Random Forest** (Alternative to ANN)
```
Ensemble of decision trees
- Good for tabular data
- Less prone to overfitting than single tree
- Fast training
- Less accurate than ANN for complex patterns
Not used in our project (we use ANN)
```

### **Support Vector Machine (SVM)** (Alternative to ANN)
```
Finds optimal hyperplane to separate classes
- Good for small datasets
- Works in high dimensions
- Computational: Slower for large data
- Not used in our project (we use ANN)
```

### **Logistic Regression** (Alternative to ANN)
```
Linear model for binary classification
- Baseline/simple approach
- Interpretable
- Cannot capture complex patterns
- Used as comparison to ANN
```

### **Convolutional Neural Networks (CNN)**
```
Specialized for image data
- Detects spatial patterns
- Efficient for images
- Not suited for tabular data
- Different from our ANN
```

### **Recurrent Neural Networks (RNN)**
```
Specialized for sequential data
- Good for time series
- Handles variable-length sequences
- Not suited for tabular data
- Different from our ANN
```

---

## Error Messages & Solutions

### **"ModuleNotFoundError: No module named 'tensorflow'"**
```
Cause: TensorFlow not installed
Solution: pip install tensorflow
```

### **"ValueError: shapes (None, 13) and (16, 13) not aligned"**
```
Cause: Input dimension mismatch
Solution: Ensure input_shape matches number of features
Check: X_train.shape[1] should equal input_shape[0]
```

### **"ResourceExhaustedError: OOM when allocating"**
```
Cause: Out of memory
Solutions:
1. Reduce batch size: 32 → 16
2. Reduce model size
3. Use GPU for more memory
4. Reduce dataset size
```

### **"Cannot assign a device for operation"**
```
Cause: GPU/device configuration issue
Solution: 
import tensorflow as tf
# Specify CPU
with tf.device('/CPU:0'):
    # your code
```

### **"Loss is NaN"**
```
Cause: Numerical instability
Solutions:
1. Reduce learning rate
2. Check for invalid data values
3. Normalize input better
4. Use gradient clipping
```

### **"Model predicts all 0s" or "all 1s"**
```
Cause: Model collapsed to single class
Solutions:
1. Check class balance
2. Adjust learning rate
3. Use class weights
4. Verify data preprocessing
```

---

## Code Snippets Cheat Sheet

### **Import Libraries**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
```

### **Load and Explore Data**
```python
# Load
df = pd.read_csv('data.csv')

# Explore
df.shape                    # (rows, columns)
df.head()                   # First rows
df.info()                   # Data types
df.describe()               # Statistics
df.isnull().sum()          # Missing values
df['target'].value_counts() # Class distribution
```

### **Train-Test Split**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% test, 80% train
    random_state=42,    # Reproducible
    stratify=y          # Preserve class distribution
)
```

### **Standardization**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)    # Fit on train
X_test_scaled = scaler.transform(X_test)          # Apply to test
```

### **Build Neural Network**
```python
model = Sequential([
    layers.Dense(16, activation='relu', input_shape=(13,)),
    layers.Dropout(0.2),
    layers.Dense(8, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
```

### **Compile Model**
```python
model.compile(
    optimizer='adam',           # Or: Adam(learning_rate=0.001)
    loss='binary_crossentropy', # For binary classification
    metrics=['accuracy']
)
```

### **Train Model**
```python
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,              # Max iterations
    batch_size=32,           # Samples per update
    validation_split=0.2,    # Monitor on 20% of train data
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5)
    ]
)
```

### **Evaluate Model**
```python
# Predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

### **Save/Load Model**
```python
# Save
model.save('my_model.h5')

# Load
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')
```

### **Visualize History**
```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()
```

---

## Quick Reference Tables

### **Activation Functions Comparison**

| Function | Formula | Range | Use | Pros | Cons |
|----------|---------|-------|-----|------|------|
| ReLU | max(0, x) | [0, ∞) | Hidden | Fast, avoids vanishing | Dead neurons |
| Sigmoid | 1/(1+e^-x) | (0, 1) | Output | Probability | Vanishing gradient |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | (-1, 1) | Hidden/Output | Better than sigmoid | Vanishing gradient |
| Softmax | e^xi/Σe^xj | (0, 1), Σ=1 | Multiclass output | Probability distribution | - |
| Linear | x | (-∞, ∞) | Output (regression) | Simple | No non-linearity |

### **Loss Functions for Different Tasks**

| Task | Loss Function | Use Case |
|------|---------------|----------|
| Binary Classification | Binary Crossentropy | Predict 0 or 1 |
| Multiclass Classification | Categorical Crossentropy | Predict multiple classes |
| Regression | Mean Squared Error (MSE) | Predict continuous values |
| Regression | Mean Absolute Error (MAE) | Robust to outliers |
| Ranking | Ranking Loss | Ranking problems |

### **Hyperparameter Tuning Guide**

| Hyperparameter | Range | Effect on Loss | Effect on Speed | Recommendation |
|---|---|---|---|---|
| Learning Rate | 0.0001-0.1 | ↓↓ if optimal | Not much | Start 0.001 |
| Epochs | 10-1000 | ↓ then ↑ (overfit) | ↑ Linear | 50-200 |
| Batch Size | 8-128 | Not much | ↑ if larger | 32 |
| Layers | 1-5 | ↓ if optimal | Slight ↑ | 2-3 |
| Neurons | 8-512 | ↓ if optimal | ↑ if more | 16-64 |
| Dropout | 0-0.5 | ↑ if high | Not much | 0.2-0.3 |

### **Metrics Interpretation Guide**

| Metric | Good Value | Interpretation |
|--------|-----------|-----------------|
| Accuracy | > 85% | % of correct predictions |
| Precision | > 0.8 | % of positive predictions correct |
| Recall | > 0.8 | % of actual positives found |
| F1-Score | > 0.8 | Balanced precision-recall |
| AUC-ROC | > 0.85 | Overall discrimination |
| Loss | < 0.3 | Prediction error |

### **Model Selection Guide**

| Problem | Model | Reason |
|---------|-------|--------|
| Tabular data, small dataset | Random Forest | Simple, effective |
| Tabular data, large dataset | Neural Network | Learns complex patterns |
| Image data | CNN | Spatial pattern detection |
| Sequential data | RNN | Temporal dependency |
| High-dimensional data | SVM | Works well in high dims |
| Interpretability important | Logistic Regression | Simple to explain |

---

## 🎯 Quick Decision Trees

### **Model Selection Flowchart**
```
Do you have image data?
├─ YES → Use CNN
└─ NO
    ├─ Do you have sequential data?
    │  ├─ YES → Use RNN
    │  └─ NO
    │      ├─ Do you need interpretability?
    │      │  ├─ YES → Use Logistic Regression/Tree
    │      │  └─ NO
    │      │      ├─ Small dataset?
    │      │      │  ├─ YES → Use SVM/Random Forest
    │      │      │  └─ NO → Use Neural Network ← OUR PROJECT
    │      │      └─ Your answer
```

### **Hyperparameter Tuning Strategy**
```
Start with defaults:
- Learning rate: 0.001
- Epochs: 100
- Batch size: 32

Train and check:
├─ Loss decreasing? Yes → Good
│  ├─ Training accuracy: 85%+? Yes → Evaluate
│  └─ No → Increase epochs, reduce learning rate
│
└─ Loss not decreasing? 
   ├─ Increase learning rate OR architecture complexity
   └─ Check if data is properly preprocessed
```

### **Debugging Low Accuracy**
```
Accuracy < 70%?
│
├─ Check Data
│  ├─ Missing values? → Handle them
│  ├─ Wrong shape? → Verify dimensions
│  └─ Not standardized? → Standardize
│
├─ Check Model
│  ├─ Too simple? → Add layers/neurons
│  ├─ Dropout too high? → Reduce (0.2 → 0.1)
│  └─ Learning rate? → Adjust (0.001 ↔ 0.01)
│
└─ Check Training
   ├─ Enough epochs? → Increase (100 → 200)
   ├─ Early stopping? → Disable for testing
   └─ Class imbalance? → Use class_weight
```

---

## 📊 Important Formulas Reference

### **Loss Calculation**
```
Binary Crossentropy:
L = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

Where:
- y = actual (0 or 1)
- ŷ = prediction (0.0 to 1.0)
```

### **Accuracy Metrics**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### **Weight Update**
```
w_new = w_old - α × (∂L/∂w)

Where:
- α = learning rate
- ∂L/∂w = gradient
```

### **Neural Network Forward Pass**
```
z = Σ(w × x) + b
a = activation(z)

Where:
- w = weight
- x = input
- b = bias
- a = activation output
```

---

## 🎓 Summary

| Concept | Key Point | Our Project |
|---------|-----------|------------|
| **ANN** | Learns patterns from data | ✓ Using it |
| **Backprop** | Updates weights via gradients | ✓ Automatic |
| **Dropout** | Prevents overfitting | ✓ 20% applied |
| **Standardization** | Normalizes feature scales | ✓ Applied |
| **Train-Test Split** | Prevents data leakage | ✓ 80-20 |
| **Early Stopping** | Stops when no improvement | ✓ Applied |
| **Accuracy** | % correct predictions | ✓ Tracked |
| **Precision** | % positive predictions correct | ✓ Tracked |
| **Recall** | % positives found | ✓ Tracked |
| **F1-Score** | Balance of precision & recall | ✓ Tracked |
| **ROC-AUC** | Overall classification quality | ✓ Tracked |

---

**Keep this glossary handy while learning! 📚**

*Last Updated: 2024*
