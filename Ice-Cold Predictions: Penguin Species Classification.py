import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
penguins = sns.load_dataset("penguins")

# Basic info
print("\n--- First 5 Rows ---\n", penguins.head())
print("\n--- Dataset Info ---")
print(penguins.info())
print("\n--- Summary Statistics ---\n", penguins.describe())

# Handle missing values
penguins = penguins.dropna()

# ============================
# Exploratory Data Analysis
# ============================

# Pairplot (species separation)
sns.pairplot(penguins, hue="species", diag_kind="kde", palette="Set2")
plt.show()

# Countplot: species by island
plt.figure(figsize=(6,4))
sns.countplot(x="island", hue="species", data=penguins, palette="Set2")
plt.title("Species Distribution by Island")
plt.show()

# Distribution of flipper length by species
plt.figure(figsize=(6,4))
sns.histplot(data=penguins, x="flipper_length_mm", hue="species", kde=True, palette="Set2")
plt.title("Flipper Length Distribution by Species")
plt.show()

# ============================
# Data Preprocessing
# ============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical columns
le = LabelEncoder()
penguins['sex'] = le.fit_transform(penguins['sex'])
penguins['island'] = le.fit_transform(penguins['island'])

# Features & Target
X = penguins.drop("species", axis=1)
y = penguins["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================
# Train Multiple ML Models
# ============================

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

# Train & Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"\n{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, preds))

    # Confusion Matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(
        confusion_matrix(y_test, preds),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=model.classes_, yticklabels=model.classes_
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ============================
# Final Model Comparison
# ============================

plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Model Comparison - Accuracy")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()
