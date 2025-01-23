
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
percentages = np.array([93.33, 86.67, 86.67, 6.67, 80, 80, 60, 73.33, 46.67, 46.67,
                        86.67, 73.33, 66.67, 93.33, 80, 66.67, 13.33, 80, 26.67, 46.67])
marks = np.array([28, 25, 22, 25, 24, 26, 16, 21, 26, 17, 27, 23, 19, 25, 23, 8, 18, 26, 14, 7])

# Reshape data for model
X = percentages.reshape(-1, 1)  # Feature
y = marks  # Target

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predictions
predicted_marks = model.predict(X)

# Results
print("Coefficients (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)
print("R-squared:", model.score(X, y))

# Plot the data and regression line
plt.scatter(percentages, marks, color='blue', label='Actual Data')
plt.plot(percentages, predicted_marks, color='red', label='Regression Line')
plt.xlabel('Percentages')
plt.ylabel('Marks')
plt.title('Linear Regression: Percentages vs Marks')
plt.legend()
plt.show()
