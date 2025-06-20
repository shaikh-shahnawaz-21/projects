# project
🔹 1. Import Libraries
python
Copy
Edit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
pandas: Dataset ko load & clean karne ke liye.

seaborn, matplotlib: Data visualization ke liye (optional).

RandomForestClassifier: Classification ke liye ML model.

accuracy_score, confusion_matrix: Performance measure.

🔹 2. Load Titanic Dataset
python
Copy
Edit
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
Titanic survivors data hai (1912 ship disaster).

Goal: Predict whether passenger survived or not (0 = No, 1 = Yes).

🔹 3. Data Cleaning
python
Copy
Edit
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
Name, Ticket, Cabin model ke liye useful nahi, isliye hata diye.

python
Copy
Edit
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
Missing Age ko average se fill kiya.

Missing Embarked (boarding location) ko most frequent value se bhara.

🔹 4. Encode Categorical Data
python
Copy
Edit
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
ML model numbers samajhta hai, text nahi.

Male/Female → 0/1

S/C/Q (ports) → 0/1/2

🔹 5. Prepare Features and Labels
python
Copy
Edit
X = df.drop('Survived', axis=1)
y = df['Survived']
X: Passenger details (input)

y: Survived or not (output)

🔹 6. Split Dataset
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
80% training, 20% testing

random_state fix karne se result repeatable rehta hai

🔹 7. Train the Random Forest Model
python
Copy
Edit
model = RandomForestClassifier(n_estimators=410)
model.fit(X_train, y_train)
Random Forest: Decision trees ka group hota hai (forest 🌳).

n_estimators=410: 410 decision trees use kiye.

🔹 8. Predict and Evaluate
python
Copy
Edit
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
Model test data pe prediction karta hai.

Accuracy: Kitna correct predict kiya.

Confusion Matrix: Class-wise details.
