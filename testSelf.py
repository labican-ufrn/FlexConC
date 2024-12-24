from selfTrainingInitial import SelfTrainingClassifier

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Carrega a base de dados Iris
data = load_iris()
X, y = data.data, data.target

# Dividi em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Converte y_train para float para aceitar NaN
y_train = y_train.astype(float)

# Simula dados não rotulados (30% dos rótulos como NaN no conjunto de treino)
rng = np.random.default_rng(42)
num_unlabeled = int(0.3 * len(y_train))
unlabeled_indices = rng.choice(len(y_train), num_unlabeled, replace=False)
y_train[unlabeled_indices] = np.nan

# Instancia e treina o modelo SelfTrainingClassifier com classificador DecisionTreeClassifier
base_classifier = DecisionTreeClassifier(random_state=42)
self_training_model = SelfTrainingClassifier(base_classifier=base_classifier, threshold=0.95, max_iter=10, criterion="threshold")
self_training_model.fit(X_train, y_train)

# Faz as previsões no conjunto de teste
y_pred = self_training_model.predict(X_test)

# Exibe as predições e rótulos reais do conjunto de teste
print("\nPredições no conjunto de teste:")
for idx, (pred, true_label) in enumerate(zip(y_pred, y_test)):
    print(f"Instância {idx}: Predição: {pred}, Rótulo Verdadeiro: {true_label}")

# Exibe a precisão do algoritmo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do SelfTrainingClassifier no conjunto de teste: {accuracy:.2f}")