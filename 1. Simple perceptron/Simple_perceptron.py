import tensorflow as tf
import numpy as np
from tensorflow import keras

print("=" * 70)
print("STUDENT EXAM PERFORMANCE PREDICTOR - PERCEPTRON NEURAL NETWORK")
print("=" * 70)
print("\nProblem: Predict if a student will PASS or FAIL based on:")
print("  - Study Hours (per day)")
print("  - Sleep Hours (per night)")
print("\n" + "=" * 70)

# Custom Training Data
# Format: [Study Hours, Sleep Hours] -> Pass(1) or Fail(0)
X_train = np.array([
    [1, 4],   # Low study, low sleep -> Fail
    [2, 5],   # Low study, average sleep -> Fail
    [3, 6],   # Medium study, good sleep -> Pass
    [4, 7],   # Good study, good sleep -> Pass
    [5, 8],   # High study, high sleep -> Pass
    [6, 7],   # High study, good sleep -> Pass
    [7, 6],   # Very high study, good sleep -> Pass
    [1, 5],   # Low study, average sleep -> Fail
    [2, 4],   # Low study, low sleep -> Fail
    [8, 7],   # Very high study, good sleep -> Pass
    [1, 8],   # Low study, high sleep -> Fail (sleeping too much, not studying)
    [3, 5],   # Medium study, average sleep -> Pass
    [4, 6],   # Good study, good sleep -> Pass
    [2, 6],   # Low study, good sleep -> Fail
    [5, 7],   # High study, good sleep -> Pass
], dtype=float)

y_train = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=float)

print("\n📚 TRAINING DATA:")
print("-" * 70)
print(f"{'Study Hours':<15} {'Sleep Hours':<15} {'Result':<15}")
print("-" * 70)
for i, (x, y) in enumerate(zip(X_train, y_train)):
    result = "PASS ✓" if y == 1 else "FAIL ✗"
    print(f"{x[0]:<15.1f} {x[1]:<15.1f} {result:<15}")
print(f"\nTotal Training Samples: {len(X_train)}")

# Create Perceptron Model
# A perceptron is the simplest neural network: just 1 neuron!
print("\n" + "=" * 70)
print("🧠 BUILDING PERCEPTRON MODEL")
print("=" * 70)

model = keras.Sequential([
    keras.layers.Dense(
        units=1,              # Single neuron (perceptron)
        activation='sigmoid', # Sigmoid for binary classification (0 or 1)
        input_shape=(2,),    # 2 inputs: study hours and sleep hours
        name='perceptron'
    )
])

print("\nModel Architecture:")
print("  Input Layer: 2 neurons (Study Hours, Sleep Hours)")
print("  Output Layer: 1 neuron (Perceptron with Sigmoid activation)")
print("  Total Parameters: 3 (2 weights + 1 bias)")

# Compile the model
model.compile(
    optimizer='adam',                    # Adaptive learning rate optimizer
    loss='binary_crossentropy',          # Loss function for binary classification
    metrics=['accuracy']
)

print("\nCompilation Settings:")
print("  Optimizer: Adam")
print("  Loss Function: Binary Cross-Entropy")
print("  Metrics: Accuracy")

# Train the model
print("\n" + "=" * 70)
print("🏋️ TRAINING THE PERCEPTRON")
print("=" * 70)

history = model.fit(
    X_train, y_train,
    epochs=100,
    verbose=0  # Silent training
)

final_loss = history.history['loss'][-1]
final_accuracy = history.history['accuracy'][-1]

print(f"\n✓ Training Complete!")
print(f"  Final Loss: {final_loss:.4f}")
print(f"  Final Accuracy: {final_accuracy:.2%}")

# Extract learned weights and bias
weights, bias = model.layers[0].get_weights()
w1, w2 = weights[0][0], weights[1][0]
b = bias[0]

print("\n" + "=" * 70)
print("⚙️ LEARNED PARAMETERS (What the Perceptron Discovered)")
print("=" * 70)
print(f"\nWeight for Study Hours (w1): {w1:.4f}")
print(f"Weight for Sleep Hours (w2):  {w2:.4f}")
print(f"Bias (b):                      {b:.4f}")

print("\n📐 Mathematical Model:")
print(f"  z = ({w1:.4f} × Study Hours) + ({w2:.4f} × Sleep Hours) + ({b:.4f})")
print(f"  output = sigmoid(z) = 1 / (1 + e^(-z))")
print(f"  prediction = PASS if output > 0.5, else FAIL")

# Test cases
print("\n" + "=" * 70)
print("🧪 TESTING THE PERCEPTRON WITH NEW DATA")
print("=" * 70)

test_cases = np.array([
    [1, 4],   # Very low study, low sleep
    [3, 7],   # Medium study, good sleep
    [6, 8],   # High study, high sleep
    [2, 3],   # Low study, very low sleep
    [7, 5],   # Very high study, average sleep
])

expected_results = ["FAIL", "PASS", "PASS", "FAIL", "PASS"]

for i, (test_input, expected) in enumerate(zip(test_cases, expected_results)):
    print(f"\n{'─' * 70}")
    print(f"TEST CASE #{i+1}")
    print(f"{'─' * 70}")
    
    study_hrs, sleep_hrs = test_input
    print(f"\n📥 Input:")
    print(f"  Study Hours: {study_hrs:.1f}")
    print(f"  Sleep Hours: {sleep_hrs:.1f}")
    
    # Step-by-step computation
    print(f"\n🔢 Step-by-Step Computation:")
    
    # Step 1: Weighted sum
    z = w1 * study_hrs + w2 * sleep_hrs + b
    print(f"\n  Step 1 - Calculate weighted sum (z):")
    print(f"    z = (w1 × Study) + (w2 × Sleep) + bias")
    print(f"    z = ({w1:.4f} × {study_hrs:.1f}) + ({w2:.4f} × {sleep_hrs:.1f}) + ({b:.4f})")
    print(f"    z = {w1*study_hrs:.4f} + {w2*sleep_hrs:.4f} + {b:.4f}")
    print(f"    z = {z:.4f}")
    
    # Step 2: Apply activation function
    sigmoid_output = 1 / (1 + np.exp(-z))
    print(f"\n  Step 2 - Apply Sigmoid activation:")
    print(f"    sigmoid(z) = 1 / (1 + e^(-z))")
    print(f"    sigmoid({z:.4f}) = 1 / (1 + e^({-z:.4f}))")
    print(f"    sigmoid({z:.4f}) = {sigmoid_output:.6f}")
    
    # Step 3: Make prediction
    prediction_value = model.predict(test_input.reshape(1, -1), verbose=0)[0][0]
    prediction = "PASS ✓" if prediction_value > 0.5 else "FAIL ✗"
    
    print(f"\n  Step 3 - Make final prediction:")
    print(f"    Output probability: {prediction_value:.6f}")
    print(f"    Threshold: 0.5")
    print(f"    Decision: {prediction_value:.6f} {'>' if prediction_value > 0.5 else '<'} 0.5")
    
    print(f"\n📤 Final Prediction: {prediction}")
    print(f"📊 Expected Result: {expected}")
    print(f"✓ {'CORRECT!' if prediction.split()[0] == expected else 'INCORRECT!'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\n🎓 The perceptron learned that:")
print(f"  • Study hours have a weight of {w1:.4f}")
print(f"  • Sleep hours have a weight of {w2:.4f}")
print(f"  • There's a baseline bias of {b:.4f}")
print("\n💡 Interpretation:")
if abs(w1) > abs(w2):
    print(f"  Study hours are MORE important (|{w1:.4f}| > |{w2:.4f}|)")
else:
    print(f"  Sleep hours are MORE important (|{w2:.4f}| > |{w1:.4f}|)")
print(f"  A student needs a good balance of both to pass!")

print("\n" + "=" * 70)