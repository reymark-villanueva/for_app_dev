import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create the dataset
data = pd.read_csv("Dataset/playgolf_data.csv")

df = pd.DataFrame(data)

# Step 2: Encode categorical features
label_encoders = {}
for column in df.columns[:-1]:  # Exclude target column
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])  # Convert text to numbers
    label_encoders[column] = le

# Step 3: Features (X) and Target (y)
X = df.drop(columns=["PlayGolf"])
y = df["PlayGolf"]

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train Naïve Bayes Model
nb = CategoricalNB()
nb.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = nb.predict(X_test)

# Step 7: Evaluate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))