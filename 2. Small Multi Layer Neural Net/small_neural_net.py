import tensorflow as tf
import numpy as np
from tensorflow import keras

print("=" * 80)
print("STUDENT GRADE PREDICTOR - MULTI-LAYER NEURAL NETWORK")
print("=" * 80)
print("\nProblem: Predict student's final grade (A, B, C, D, F) based on:")
print("  - Study Hours (per day)")
print("  - Sleep Hours (per night)")
print("  - Assignment Score (0-100)")
print("\n📊 Grade Scale: A(90-100), B(80-89), C(70-79), D(60-69), F(0-59)")
print("\n" + "=" * 80)

# Custom Training Data - More complex patterns
# Format: [Study Hours, Sleep Hours, Assignment Score] -> Grade (0=F, 1=D, 2=C, 3=B, 4=A)
X_train = np.array([
    [8, 7, 95],   # A
    [7, 8, 92],   # A
    [6, 7, 88],   # B
    [5, 6, 85],   # B
    [4, 6, 78],   # C
    [3, 5, 75],   # C
    [2, 4, 65],   # D
    [2, 5, 68],   # D
    [1, 4, 45],   # F
    [1, 3, 50],   # F
    [7, 7, 90],   # A
    [6, 6, 82],   # B
    [5, 7, 88],   # B
    [4, 5, 72],   # C
    [3, 6, 76],   # C
    [2, 3, 62],   # D
    [1, 5, 55],   # F
    [8, 8, 98],   # A
    [5, 5, 80],   # B
    [3, 4, 70],   # C
    [7, 6, 93],   # A
    [4, 7, 84],   # B
    [2, 6, 67],   # D
    [6, 8, 91],   # A
    [3, 3, 58],   # F
], dtype=float)

# Grade labels: 0=F, 1=D, 2=C, 3=B, 4=A
y_train = np.array([4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 4, 3, 3, 2, 2, 1, 0, 4, 3, 2, 4, 3, 1, 4, 0])

# Normalize input features for better training
X_train_normalized = X_train / np.array([10.0, 10.0, 100.0])  # Scale to 0-1 range

grade_names = ['F', 'D', 'C', 'B', 'A']

print("\n📚 TRAINING DATA SAMPLE (First 10 samples):")
print("-" * 80)
print(f"{'Study Hrs':<12} {'Sleep Hrs':<12} {'Assignment':<12} {'Grade':<12}")
print("-" * 80)
for i in range(min(10, len(X_train))):
    print(f"{X_train[i][0]:<12.1f} {X_train[i][1]:<12.1f} {X_train[i][2]:<12.1f} {grade_names[y_train[i]]:<12}")
print(f"\n... and {len(X_train) - 10} more samples")
print(f"Total Training Samples: {len(X_train)}")

# Create Multi-Layer Neural Network
print("\n" + "=" * 80)
print("🧠 BUILDING MULTI-LAYER NEURAL NETWORK")
print("=" * 80)

model = keras.Sequential([
    # Input layer (implicit) - 3 features
    
    # Hidden Layer 1 - Learn complex patterns
    keras.layers.Dense(
        units=8,
        activation='relu',
        input_shape=(3,),
        name='hidden_layer_1'
    ),
    
    # Hidden Layer 2 - Learn higher-level features
    keras.layers.Dense(
        units=6,
        activation='relu',
        name='hidden_layer_2'
    ),
    
    # Output Layer - 5 neurons for 5 grades (F, D, C, B, A)
    keras.layers.Dense(
        units=5,
        activation='softmax',  # Softmax for multi-class classification
        name='output_layer'
    )
])

print("\n📐 Model Architecture:")
print("-" * 80)
print("  Layer 1 (Input):          3 neurons  → [Study, Sleep, Assignment]")
print("  Layer 2 (Hidden 1):       8 neurons  → ReLU activation")
print("  Layer 3 (Hidden 2):       6 neurons  → ReLU activation")
print("  Layer 4 (Output):         5 neurons  → Softmax activation [F,D,C,B,A]")
print("-" * 80)

# Count parameters
layer1_params = (3 * 8) + 8  # weights + biases
layer2_params = (8 * 6) + 6
layer3_params = (6 * 5) + 5
total_params = layer1_params + layer2_params + layer3_params

print(f"\n  Hidden Layer 1 Parameters: {layer1_params} (3×8 weights + 8 biases)")
print(f"  Hidden Layer 2 Parameters: {layer2_params} (8×6 weights + 6 biases)")
print(f"  Output Layer Parameters:   {layer3_params} (6×5 weights + 5 biases)")
print(f"  Total Parameters:          {total_params}")

print("\n🔍 Key Differences from Perceptron:")
print("  ✓ Multiple hidden layers (2 vs 0)")
print("  ✓ Multiple neurons per layer (8, 6 vs 1)")
print("  ✓ ReLU activation in hidden layers (vs none)")
print("  ✓ Softmax output for multi-class (vs sigmoid for binary)")
print(f"  ✓ Many more parameters ({total_params} vs 3)")

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # For integer labels
    metrics=['accuracy']
)

print("\nCompilation Settings:")
print("  Optimizer: Adam (adaptive learning rate)")
print("  Loss: Sparse Categorical Cross-Entropy (multi-class)")
print("  Metrics: Accuracy")

# Train the model
print("\n" + "=" * 80)
print("🏋️ TRAINING THE NEURAL NETWORK")
print("=" * 80)

history = model.fit(
    X_train_normalized, y_train,
    epochs=150,
    verbose=0,
    batch_size=5
)

final_loss = history.history['loss'][-1]
final_accuracy = history.history['accuracy'][-1]

print(f"\n✓ Training Complete!")
print(f"  Epochs: 150")
print(f"  Final Loss: {final_loss:.4f}")
print(f"  Final Accuracy: {final_accuracy:.2%}")

# Test cases
print("\n" + "=" * 80)
print("🧪 TESTING THE NEURAL NETWORK WITH NEW DATA")
print("=" * 80)

test_cases = np.array([
    [8, 8, 96],   # Excellent everything
    [5, 6, 83],   # Good study, good assignment
    [3, 5, 74],   # Average
    [2, 4, 63],   # Below average
    [1, 3, 48],   # Poor performance
    [6, 7, 77],   # Good study but average assignment
])

expected_grades = ["A", "B", "C", "D", "F", "C"]

test_cases_normalized = test_cases / np.array([10.0, 10.0, 100.0])

for i, (test_input, test_norm, expected) in enumerate(zip(test_cases, test_cases_normalized, expected_grades)):
    print(f"\n{'═' * 80}")
    print(f"TEST CASE #{i+1}")
    print(f"{'═' * 80}")
    
    study, sleep, assignment = test_input
    print(f"\n📥 Input Features:")
    print(f"  Study Hours:      {study:.1f}")
    print(f"  Sleep Hours:      {sleep:.1f}")
    print(f"  Assignment Score: {assignment:.1f}")
    
    print(f"\n🔢 Neural Network Processing:")
    print(f"{'─' * 80}")
    
    # Use the functional approach that works in Keras 3.x
    input_data = test_norm.reshape(1, -1)
    
    # Get layer outputs by calling layers directly
    layer1_output = model.layers[0](input_data).numpy()[0]
    
    print(f"\n  Step 1 - Hidden Layer 1 (8 neurons with ReLU):")
    print(f"    Input (normalized): [{test_norm[0]:.3f}, {test_norm[1]:.3f}, {test_norm[2]:.3f}]")
    print(f"    Output activations: {layer1_output[:4].round(3)} ... (showing first 4)")
    print(f"    Active neurons: {np.sum(layer1_output > 0)}/8")
    
    # Layer 2 output
    x = model.layers[0](input_data)
    layer2_output = model.layers[1](x).numpy()[0]
    
    print(f"\n  Step 2 - Hidden Layer 2 (6 neurons with ReLU):")
    print(f"    Output activations: {layer2_output[:4].round(3)} ... (showing first 4)")
    print(f"    Active neurons: {np.sum(layer2_output > 0)}/6")
    
    # Final prediction
    predictions = model.predict(test_norm.reshape(1, -1), verbose=0)[0]
    predicted_grade_idx = np.argmax(predictions)
    predicted_grade = grade_names[predicted_grade_idx]
    confidence = predictions[predicted_grade_idx] * 100
    
    print(f"\n  Step 3 - Output Layer (5 neurons with Softmax):")
    print(f"    Raw probability distribution:")
    for j, (grade, prob) in enumerate(zip(grade_names, predictions)):
        bar_length = int(prob * 40)
        bar = '█' * bar_length
        marker = ' ←' if j == predicted_grade_idx else ''
        print(f"      {grade}: {prob:.4f} ({prob*100:5.2f}%) {bar}{marker}")
    
    print(f"\n📤 Final Prediction:")
    print(f"  Predicted Grade: {predicted_grade}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"  Expected Grade: {expected}")
    print(f"  Result: {'✓ CORRECT!' if predicted_grade == expected else '✗ INCORRECT'}")

print("\n" + "=" * 80)
print("📊 NETWORK INSIGHTS")
print("=" * 80)

print("\n💡 How Multi-Layer Networks Learn Better:")
print("  • Layer 1: Detects basic patterns (e.g., 'high study', 'low sleep')")
print("  • Layer 2: Combines patterns (e.g., 'works hard but tired')")
print("  • Output: Maps complex patterns to specific grades")
print("\n  This allows the network to learn non-linear relationships")
print("  that a single perceptron cannot capture!")

print(f"\n📈 Training Performance:")
print(f"  Final Accuracy: {final_accuracy:.2%}")
print(f"  Final Loss: {final_loss:.4f}")

print("\n" + "=" * 80)