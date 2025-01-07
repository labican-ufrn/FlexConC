from abc import abstractmethod
from typing import Dict, List, Optional
import numpy as np
from sklearn.base import clone
from sklearn.utils import safe_mask
from sklearn.metrics import accuracy_score
from selfTrainingInitial import SelfTrainingClassifier


class BaseFlexCon(SelfTrainingClassifier):
    def __init__(self, base_classifier, threshold=0.95, verbose=False, max_iter=10):
        self.base_classifier = base_classifier
        self.threshold = threshold
        self.verbose = verbose
        self.max_iter = max_iter
        self.pred_x_it = {}
        self.cl_memory = []
        self.classifier_ = clone(base_classifier)
        self.accuracy_history = []

    @abstractmethod
    def adjust_threshold(self, local_measure):
        """
        Método abstrato para ajuste do threshold. Cada classe derivada deve implementar sua lógica específica.
        """
        pass

    def fit(self, X, y):
        # Inicialização do classificador e acurácia inicial
        labeled_indices = np.where(y != -1)[0]
        unlabeled_indices = np.where(y == -1)[0]

        init_acc = self.train_new_classifier(labeled_indices, X, y)

        for self.n_iter_ in range(self.max_iter):

            # Verifica se ainda há instâncias não rotuladas
            if len(unlabeled_indices) == 0:
                break

            # Fazer previsões e selecionar instâncias
            self.pred_x_it = self.storage_predict(
                unlabeled_indices,
                self.classifier_.predict_proba(X[unlabeled_indices]).max(axis=1),
                self.classifier_.predict(X[unlabeled_indices])
            )
            selected_indices, predictions = self.select_instances_by_rules()

            if len(selected_indices) == 0:
                break

            # Atualizar o conjunto de instâncias rotuladas
            self.add_new_labeled(selected_indices, selected_indices, predictions)

            # Calcula a métrica local (usada para ajuste do threshold)
            local_measure = self.calc_local_measure(X[safe_mask(X, labeled_indices)], y[labeled_indices], self.classifier_)

            # Atualiza o histórico de precisão
            self.accuracy_history.append(local_measure)

            # Ajuste do threshold baseado em critério específico da subclasse
            self.adjust_threshold(local_measure)

            # Re-treinar o classificador com o novo conjunto de dados rotulados
            init_acc = self.train_new_classifier(labeled_indices, X, y)

        return self

    def calc_local_measure(self, X, y_true, classifier):
        """
        Calcula o valor da acurácia do modelo

        Args:
            X: instâncias
            y_true: classes
            classifier: modelo

        Returns:
            Retorna a acurácia do modelo
        """
        y_pred = classifier.predict(X)
        return accuracy_score(y_true, y_pred)

    def update_memory(self, instances: List, labels: List, weights: Optional[List] = None):
        """
        Atualiza a matriz de instâncias rotuladas

        Args:
            instances: instâncias
            labels: rotulos
            weights: Pesos de cada classe
        """
        if not weights:
            weights = [1 for _ in range(len(instances))]
        for instance, label, weight in zip(instances, labels, weights):
            self.cl_memory[instance][label] += weight

    def remember(self, X: List) -> List:
        """
        Responsável por armazenar como está as instâncias dado um  momento no
        código

        Args:
            X: Lista com as instâncias

        Returns:
            A lista memorizada em um dado momento
        """
        return [np.argmax(self.cl_memory[x]) for x in X]

    def storage_predict(self, idx, confidence, classes) -> Dict[int, Dict[float, int]]:
        """
        Responsável por armazenar o dicionário de dados da matriz

        Args:
            idx: indices de cada instância
            confidence: taxa de confiança para a classe destinada
            classes: indices das classes

        Returns:
            Retorna o dicionário com as classes das instâncias não rotuladas
        """
        memo = {}
        for i, conf, label in zip(idx, confidence, classes):
            memo[i] = {"confidence": conf, "classes": label}
        return memo

    def rule_1(self):
        """
        Regra responsável por verificar se as classes são iguais E ambas as
        confianças preditas são maiores que o limiar.

        Returns:
            A lista correspondente pela condição.
        """
        selected, classes_selected = [], []

        for i, data in self.pred_x_it.items():
            first_confidence = self.dict_first[i]["confidence"]
            confidence = data["confidence"]
            first_class = self.dict_first[i]["classes"]
            predicted_class = data["classes"]

            if first_confidence >= self.threshold and confidence >= self.threshold and first_class == predicted_class:
                selected.append(i)
                classes_selected.append(predicted_class)

        return selected, classes_selected

    def rule_2(self):
        """
        Regra responsável por verificar se as classes são iguais E ao menos uma
        das confianças preditas é maior que o limiar.

        Returns:
            A lista correspondente pela condição.
        """
        selected, classes_selected = [], []

        for i, data in self.pred_x_it.items():
            first_confidence = self.dict_first[i]["confidence"]
            confidence = data["confidence"]
            first_class = self.dict_first[i]["classes"]
            predicted_class = data["classes"]

            if (first_confidence >= self.threshold or confidence >= self.threshold) and first_class == predicted_class:
                selected.append(i)
                classes_selected.append(predicted_class)

        return selected, classes_selected

    def rule_3(self):
        """
        Regra responsável por verificar se as classes são diferentes E ambas as
        confianças preditas são maiores que o limiar.

        Returns:
            A lista correspondente pela condição.
        """
        selected = []

        for i, data in self.pred_x_it.items():
            first_confidence = self.dict_first[i]["confidence"]
            confidence = data["confidence"]
            first_class = self.dict_first[i]["classes"]
            predicted_class = data["classes"]

            if first_class != predicted_class and first_confidence >= self.threshold and confidence >= self.threshold:
                selected.append(i)

        return selected, self.remember(selected)

    def rule_4(self):
        """
        Regra responsável por verificar se as classes são diferentes E uma das
        confianças preditas é maior que o limiar.

        Returns:
            A lista correspondente pela condição.
        """
        selected = []

        for i, data in self.pred_x_it.items():
            first_confidence = self.dict_first[i]["confidence"]
            confidence = data["confidence"]
            first_class = self.dict_first[i]["classes"]
            predicted_class = data["classes"]

            if first_class != predicted_class and (first_confidence >= self.threshold or confidence >= self.threshold):
                selected.append(i)

        return selected, self.remember(selected)

    def train_new_classifier(self, has_label, X, y):
        """
        Treina um classificador usando apenas as instâncias rotuladas e mede sua acurácia.

        Args:
            has_label: índices das instâncias rotuladas
            X: instâncias
            y: rótulos

        Returns:
            Acurácia inicial do modelo
        """

        # Inicializa as estruturas de transdução e índices de iteração para as instâncias rotuladas
        self.transduction_ = np.copy(y)
        self.labeled_iter_ = np.full_like(y, -1)
        self.labeled_iter_[has_label] = 0
        self.init_labeled_ = has_label.copy()

        # Clona e treina o classificador base com dados rotulados
        self.classifier_ = clone(self.base_classifier)
        labeled_mask = ~np.isnan(y) & (y != -1)
        X_labeled = X[labeled_mask]
        y_labeled = self.transduction_[labeled_mask]
        self.classifier_.fit(X_labeled, y_labeled)

        # Calcula e retorna a acurácia inicial do modelo com as instâncias rotuladas
        init_acc = self.calc_local_measure(X_labeled, y_labeled, self.classifier_)
        return init_acc



    def add_new_labeled(self, selected_full, selected, pred):
        """
        Função que retorna as intâncias rotuladas

        Args:
            selected_full: lista com os indices das instâncias originais
            selected: lista das intâncias com acc acima do limiar
            pred: predição das instâncias não rotuladas
        """
        self.transduction_[selected_full] = pred[selected]
        self.labeled_iter_[selected_full] = self.n_iter_

    def select_instances_by_rules(self):
        """
        Função responsável por gerenciar todas as regras de inclusão do método

        Returns:
            _type_: _description_
        """
        insertion_rules = [self.rule_1, self.rule_2, self.rule_3, self.rule_4]

        for rule in insertion_rules:
            selected, pred = rule()

            if selected:
                return np.array(selected), pred
        return np.array([]), ""
