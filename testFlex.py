import numpy as np
from flexconc import *
from mlabican import FlexCon, FlexConG
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Carrega a base de dados Iris
data = load_iris()
X, y = data.data, data.target

# Divide em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Converte y_train para float para aceitar NaN
y_train = y_train.astype(float)

# Simula dados não rotulados (30% dos rótulos como NaN no conjunto de treino)
rng = np.random.default_rng(42)
num_unlabeled = int(0.3 * len(y_train))
unlabeled_indices = rng.choice(len(y_train), num_unlabeled, replace=False)
y_train[unlabeled_indices] = np.nan

# Função para treinar e avaliar um modelo FlexCon
def train_and_evaluate_flexcon_model(model, X_train, y_train, X_test, y_test):
    # Treina o modelo
    model.fit(X_train, y_train)

    # Faz as previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Exibe as predições e rótulos reais do conjunto de teste
    print("\nPredições no conjunto de teste:")
    for idx, (pred, true_label) in enumerate(zip(y_pred, y_test)):
        print(f"Instância {idx}: Predição: {pred}, Rótulo Verdadeiro: {true_label}")

    # Exibe a precisão do algoritmo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia no conjunto de teste: {accuracy:.2f}")

# Inicializa o classificador base
base_classifier = DecisionTreeClassifier(random_state=42)

# Teste do FlexCon-G
print("Teste do FlexCon-G:")
flexcon_g_model = FlexConG(base_classifier=base_classifier, cr=0.05, threshold=0.95, verbose=True)
train_and_evaluate_flexcon_model(flexcon_g_model, X_train, y_train, X_test, y_test)

# Teste do FlexCon
print("\nTeste do FlexCon:")
flexcon_model = FlexCon(base_classifier=base_classifier, threshold=0.95, verbose=True)
train_and_evaluate_flexcon_model(flexcon_model, X_train, y_train, X_test, y_test)

# Teste do FlexCon-C1s
print("\nTeste do FlexCon-C1s:")
flexcon_c1s_model = FlexConC(base_classifier=base_classifier, select_model=flexconC1s, cr=0.05, threshold=0.95, verbose=True)
train_and_evaluate_flexcon_model(flexcon_c1s_model, X_train, y_train, X_test, y_test)

# Teste do FlexCon-C1v
print("\nTeste do FlexCon-C1v:")
flexcon_c1v_model = FlexConC(base_classifier=base_classifier, select_model=flexconC1v, cr=0.05, threshold=0.95, verbose=True)
train_and_evaluate_flexcon_model(flexcon_c1v_model, X_train, y_train, X_test, y_test)

# Teste do FlexCon-C2
print("\nTeste do FlexCon-C2:")
flexcon_c2_model = FlexConC(base_classifier=base_classifier, select_model=flexconC2, cr=0.05, threshold=0.95, verbose=True)
train_and_evaluate_flexcon_model(flexcon_c2_model, X_train, y_train, X_test, y_test)
