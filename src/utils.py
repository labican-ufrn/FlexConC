from statistics import mean, stdev

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as Tree

# LISTA DE CLASSIFICADORES
list_tree = [
    Tree(),
    Tree(splitter="random"),
    Tree(max_features=None),
    Tree(criterion="entropy"),
    Tree(criterion="entropy", splitter="random"),
    Tree(criterion="entropy", max_features=None),
    Tree(criterion="entropy", max_features=None, splitter="random"),
    Tree(criterion="entropy", max_features='sqrt', splitter="random"),
    Tree(max_features='sqrt', splitter="random"),
    Tree(max_features=None, splitter="random")]

list_knn_Prelax = [
    KNN(n_neighbors=4, weights='distance'), KNN(n_neighbors=4),
    KNN(n_neighbors=5, weights='distance'), KNN(n_neighbors=5),
    KNN(n_neighbors=5, weights='distance'), KNN(n_neighbors=5),
    KNN(n_neighbors=5, weights='distance'), KNN(n_neighbors=5),
    KNN(n_neighbors=5, weights='distance'), KNN(n_neighbors=5)
]

list_knn_seeds = [
    KNN(n_neighbors=4, weights='distance'), KNN(n_neighbors=4),
    KNN(n_neighbors=5, weights='distance'), KNN(n_neighbors=5),
    KNN(n_neighbors=6, weights='distance'), KNN(n_neighbors=6),
    KNN(n_neighbors=6, weights='distance'), KNN(n_neighbors=6),
    KNN(n_neighbors=6, weights='distance'), KNN(n_neighbors=6)
]

list_knn_full = [
    KNN(n_neighbors=4, weights='distance'), KNN(n_neighbors=4),
    KNN(n_neighbors=5, weights='distance'), KNN(n_neighbors=5),
    KNN(n_neighbors=6, weights='distance'), KNN(n_neighbors=6),
    KNN(n_neighbors=7, weights='distance'), KNN(n_neighbors=7),
    KNN(n_neighbors=8, weights='distance'), KNN(n_neighbors=8)
]

def select_labels(y_train, X_train, labelled_percentage):
    """
    Responsável por converter o array de rótulos das instâncias com base
    nas instâncias selecionadas randomicamente.

    Args:
        - y_train (Array): Classes usadas no treinamento
        - X_train (Array): Instâncias
        - labelled_percentage (float): % de instâncias que ficarão com o
            rótulo.

    Returns:
        Retorna o array de classes com base nos rótulos das instância
        selecionadas.
    """
    class_dist = np.bincount(y_train)
    min_acceptable = np.trunc(class_dist * labelled_percentage)
    instances = []

    for lab, cls_dist in enumerate(min_acceptable):
        instances += np.random.choice(
            np.where(y_train == lab)[0],
            int(cls_dist) or 1,
            replace=False
        ).tolist()

    mask = np.ones(len(X_train), bool)
    mask[instances] = 0
    y_train[mask] = -1
    return y_train

def result(option, dataset, y_test, y_pred, path, labelled_level, rounds):
    """
    Salva os resultados dos comitês em arquivos CSV com base na opção escolhida.

    Args:
        option (int): Identificador do comitê.
        dataset (str): Nome do dataset.
        y_test (list): Rótulos verdadeiros para teste.
        y_pred (list): Rótulos previstos pelo modelo.
        path (str): Caminho do diretório para salvar os arquivos.
        labelled_level (float): Percentual de dados rotulados na iteração.
        rounds (int): Número da rodada atual.

    Returns:
        float: F1-Score (macro) multiplicado por 100.
    """
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    committee_files = {
        1: "Comite_Naive_.csv",
        2: "Comite_Tree_.csv",
        3: "Comite_KNN_.csv",
        4: "Comite_Heterogeneo_.csv",
    }

    file_name = committee_files.get(option, "Comite_Heterogeneo_.csv")

    with open(f'{path}/{file_name}', 'a', encoding='utf-8') as f:
        f.write(f'\n{rounds},"{dataset}",{labelled_level},{acc},{f1}')

    return f1 * 100

def calculate_mean_stdev(
    fold_result_acc,
    option,
    labelled_level,
    path,
    dataset,
    fold_result_f1_score
):
    """
    Calcula e salva a média e o desvio padrão de ACC e F1-Score para diferentes comitês.

    Args:
        fold_result_acc (list): Lista de ACCs por rodada.
        option (int): Identificador do comitê.
        labelled_level (float): Percentual de dados rotulados.
        path (str): Caminho do diretório para salvar os arquivos.
        dataset (str): Nome do dataset.
        fold_result_f1_score (list): Lista de F1-Scores por rodada.
    """
    acc_average = mean(fold_result_acc)
    standard_deviation_acc = stdev(fold_result_acc)
    f1_average = mean(fold_result_f1_score)
    standard_deviation_f1 = stdev(fold_result_f1_score)

    committee_files = {
        1: "Comite_Naive_F.csv",
        2: "Comite_Tree_F.csv",
        3: "Comite_KNN_F.csv",
        4: "Comite_Heterogeneo_F.csv",
    }

    file_name = committee_files.get(option, "Comite_Heterogeneo_F.csv")

    with open(f'{path}/{file_name}', 'a', encoding='utf-8') as f:
        f.write(
            f'\n"{dataset}",{labelled_level},{acc_average},{standard_deviation_acc},{f1_average},{standard_deviation_f1}'
        )
