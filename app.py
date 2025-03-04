import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
num_samples = 5000

data = {
    "Temperature": np.random.normal(70, 15, num_samples),
    "Vibration": np.random.normal(5, 2, num_samples),
    "Pressure": np.random.normal(100, 20, num_samples),
    "Operating_Hours": np.random.randint(100, 10000, num_samples),
}

failure_prob = (
    0.3 * (data["Temperature"] > 85) +
    0.3 * (data["Vibration"] > 7) +
    0.2 * (data["Pressure"] > 120) +
    0.2 * (data["Operating_Hours"] > 8000)
)
data["Failure"] = np.where(failure_prob > np.random.rand(num_samples), 1, 0)

df = pd.DataFrame(data)

X = df.drop(columns=["Failure"])
y = df["Failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Failure Prediction")
plt.show()
