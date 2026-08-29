from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Breast Cancer Wisconsin dataset
cancer = datasets.load_breast_cancer()
X = cancer.data        # 30 features (mean radius, texture, perimeter, etc.)
y = cancer.target      # Binary classification: 0 (Malignant), 1 (Benign)

# 2. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Scale the features (Crucial to prevent larger features from dominating)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Instantiate and Train the SVM Model
# We use the RBF kernel and explicitly set a soft-margin parameter (C)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 5. Make Predictions and Evaluate
y_pred = svm_model.predict(X_test_scaled)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
