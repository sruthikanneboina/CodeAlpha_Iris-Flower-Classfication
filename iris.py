import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import joblib

# Load the dataset
df = pd.read_csv("Iris.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Drop unnecessary columns
df = df.drop(columns=["Id"])

# Prepare features and labels
X = df.drop(columns=["Species"])
y = df["Species"]

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("📄 Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Simplify species labels for plots
df["Species"] = df["Species"].str.replace("Iris-", "")

# Pairplot to visualize features
sns.pairplot(df, hue="Species")
plt.suptitle("Iris Feature Distributions", y=1.02)
plt.show()

# Save the model
joblib.dump(model, "iris_model.pkl")
print("✅ Model saved as iris_model.pkl")
