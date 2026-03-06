# =============================================================
#   LINEAR REGRESSION WITH SCIKIT-LEARN
#   Based on Andrew Ng's ML Specialization — Course 1
#   Example: Predicting House Prices
# =============================================================

# ── STEP 0: Import Libraries ──────────────────────────────────
import numpy as np                              # For numerical arrays
import matplotlib.pyplot as plt                 # For plotting graphs
from sklearn.linear_model import LinearRegression  # The model
from sklearn.model_selection import train_test_split  # To split data
from sklearn.metrics import mean_squared_error  # To evaluate model
from sklearn.preprocessing import StandardScaler   # For feature scaling


# =============================================================
# STEP 1: PREPARE THE DATA
# =============================================================
# We have house sizes (in 100 sq ft) and their prices (in $100k)
# This is our training data — (input x, correct output y)

# Feature: House Size (x)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
# reshape(-1, 1) converts a 1D array [1,2,3...] into a 2D column
# Scikit-learn always expects X to be 2D — even for 1 feature

# Target: House Price (y)
y = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
# True relationship here is: y = 2x + 1
# The model doesn't know this — it will learn it from data


# =============================================================
# STEP 2: SPLIT DATA INTO TRAINING SET AND TEST SET
# =============================================================
# We never evaluate the model on data it was trained on
# Training set → model learns from this
# Test set     → we evaluate how well model generalizes

X_train, X_test, y_train, y_test = train_test_split(
    X,              # Features
    y,              # Targets
    test_size=0.2,  # 20% of data goes to test set (2 houses)
    random_state=42 # Fixed seed so results are reproducible
)

print("=" * 50)
print("STEP 2: DATA SPLIT")
print("=" * 50)
print(f"Training samples : {X_train.shape[0]}")  # 8 houses
print(f"Test samples     : {X_test.shape[0]}")   # 2 houses


# =============================================================
# STEP 3: FEATURE SCALING (Z-score Normalization)
# =============================================================
# As Andrew Ng teaches — scale features before training
# StandardScaler applies: x_scaled = (x - mean) / std_dev

scaler = StandardScaler()

# IMPORTANT: Fit the scaler ONLY on training data
# (compute mean and std from training set only)
X_train_scaled = scaler.fit_transform(X_train)
# fit_transform = compute mean/std AND apply scaling in one step

# Apply the SAME scaling to test data
# (use training mean/std — never refit on test data)
X_test_scaled = scaler.transform(X_test)
# transform only = apply existing mean/std (no recomputing)

print("\n" + "=" * 50)
print("STEP 3: FEATURE SCALING")
print("=" * 50)
print(f"Original X_train (first 3): {X_train[:3].flatten()}")
print(f"Scaled   X_train (first 3): {X_train_scaled[:3].flatten().round(3)}")


# =============================================================
# STEP 4: CREATE AND TRAIN THE MODEL
# =============================================================
# LinearRegression() internally finds the best w and b
# that minimize the cost function J(w,b)

model = LinearRegression()
# At this point: model exists but w and b are not learned yet

model.fit(X_train_scaled, y_train)
# .fit() = TRAINING
# This is where the model finds the optimal w and b
# by minimizing J(w,b) = (1/2m) * sum(y_hat - y)^2

print("\n" + "=" * 50)
print("STEP 4: MODEL TRAINED")
print("=" * 50)
print(f"Learned weight w (slope)     : {model.coef_[0]:.4f}")
# model.coef_ = the learned w parameter
# This is the slope of the line f(x) = wx + b

print(f"Learned bias   b (intercept) : {model.intercept_:.4f}")
# model.intercept_ = the learned b parameter
# This is where the line crosses the y-axis


# =============================================================
# STEP 5: MAKE PREDICTIONS
# =============================================================
# Use the trained model to predict prices on the test set

y_pred = model.predict(X_test_scaled)
# model.predict() computes: y_hat = w * x_scaled + b
# for each house in the test set

print("\n" + "=" * 50)
print("STEP 5: PREDICTIONS ON TEST SET")
print("=" * 50)
print(f"{'House Size':>12} | {'Actual Price':>13} | {'Predicted Price':>16}")
print("-" * 47)
for size, actual, predicted in zip(X_test.flatten(), y_test, y_pred):
    print(f"{size:>10.0f}   |{actual:>12.1f}   |{predicted:>15.2f}")


# =============================================================
# STEP 6: EVALUATE THE MODEL
# =============================================================
# Cost Function (MSE) tells us how wrong our predictions are
# Lower MSE = better model

mse = mean_squared_error(y_test, y_pred)
# MSE = (1/m) * sum(y_pred - y_actual)^2
# Note: sklearn's MSE doesn't include the 1/2 factor Andrew Ng uses

rmse = np.sqrt(mse)
# RMSE = root mean squared error
# Same unit as y — easier to interpret

print("\n" + "=" * 50)
print("STEP 6: MODEL EVALUATION")
print("=" * 50)
print(f"Mean Squared Error  (MSE)  : {mse:.4f}")
print(f"Root Mean Sq. Error (RMSE) : {rmse:.4f}")
# RMSE tells you: on average, predictions are off by RMSE units


# =============================================================
# STEP 7: PREDICT ON A NEW HOUSE (Real-world usage)
# =============================================================
# A new house comes in — size = 5.5 (550 sq ft)
# We must scale it using the SAME scaler from training

new_house_size = np.array([[5.5]])          # Must be 2D for sklearn
new_house_scaled = scaler.transform(new_house_size)  # Scale it
predicted_price  = model.predict(new_house_scaled)   # Predict

print("\n" + "=" * 50)
print("STEP 7: PREDICT NEW HOUSE")
print("=" * 50)
print(f"New House Size        : 300 sq ft")
print(f"Predicted Price       : ${predicted_price[0]:.2f} × $100k = ${predicted_price[0]*100:.0f}k")


# =============================================================
# STEP 8: VISUALIZE EVERYTHING
# =============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Linear Regression — House Price Prediction", fontsize=14, fontweight='bold')

# ── Plot 1: Regression Line ───────────────────────────────────
ax1 = axes[0]

# Plot training data points
ax1.scatter(X_train, y_train,
            color='blue', label='Training Data', zorder=5, s=80)

# Plot test data points
ax1.scatter(X_test, y_test,
            color='green', label='Test Data', zorder=5, s=80, marker='s')

# Draw the regression line (best fit line the model learned)
X_line = np.linspace(1, 10, 100).reshape(-1, 1)   # 100 points from 1 to 10
X_line_scaled = scaler.transform(X_line)            # Scale them
y_line = model.predict(X_line_scaled)               # Predict for line

ax1.plot(X_line, y_line,
         color='red', linewidth=2, label=f'Regression Line: ŷ = {model.coef_[0]:.1f}x + {model.intercept_:.1f}')

ax1.set_xlabel("House Size (100 sq ft)")
ax1.set_ylabel("Price ($100k)")
ax1.set_title("Fitted Regression Line")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Plot 2: Actual vs Predicted ───────────────────────────────
ax2 = axes[1]

# Predict on all training data to compare actual vs predicted
y_train_pred = model.predict(X_train_scaled)

# Perfect prediction = points on the diagonal line
ax2.scatter(y_train, y_train_pred,
            color='blue', label='Training', s=80)
ax2.scatter(y_test, y_pred,
            color='green', label='Test', s=80, marker='s')

# Draw the perfect prediction line (y = x)
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val],
         'r--', linewidth=2, label='Perfect Prediction')
# Points ON this line = prediction was exactly right
# Points OFF this line = prediction had some error

ax2.set_xlabel("Actual Price ($100k)")
ax2.set_ylabel("Predicted Price ($100k)")
ax2.set_title("Actual vs Predicted Prices")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved!")
