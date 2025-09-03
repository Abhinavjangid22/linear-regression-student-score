import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("../data/student_scores.csv")
print("Dataset:\n", df.head())

# Features and Target
X = df[["StudyHours"]]
y = df["TestScore"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot Best Fit Line
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Study Hours")
plt.ylabel("Test Score")
plt.title("Study Hours vs Test Score")
plt.show()

# Custom prediction
hours = 7.5
predicted_score = model.predict([[hours]])
print(f"If a student studies {hours} hours, predicted score = {predicted_score[0]:.2f}")
