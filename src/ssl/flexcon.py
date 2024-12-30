from abc import abstractmethod
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils import safe_mask


class BaseFlexConC(SelfTrainingClassifier):
    """
    Funcão do Flexcon-C, responsável por classificar instâncias com base em
        modelos de aprendizado semisupervisionado
    """

    def __init__(self, base_estimator, cr=0.05, threshold=0.95, verbose=False):
        super().__init__(
            base_estimator=base_estimator, threshold=threshold, max_iter=100
        )
        self.cr: float = cr
        self.threshold: float = threshold
        self.verbose = verbose
        self.old_selected: List = []
        self.dict_first: Dict = {}
        self.init_labeled_: List = []
        self.labeled_iter_: List = []
        self.transduction_: List = []
        self.classes_: List = []
        self.termination_condition_ = ""
        self.pred_x_it: Dict = {}
        self.cl_memory: List = []
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_select_ = clone(self.base_estimator)

    @abstractmethod
    def fit(self, X, y):
        ...

    def validate(self):
        """
            Validate the fitted estimator since `predict_proba` can be
            delegated to an underlying "final" fitted estimator as
            generally done in meta-estimator or pipeline.
        """
        try:
            if not hasattr(self.base_estimator, "predict_proba"):
                msg = "base_estimator ({}) should implement predict_proba!"
                raise ValueError(msg.format(type(self.base_estimator).__name__))
        except ValueError:
            return False
        return True

    def calc_local_measure(self, X, y_true, classifier):
        """ge
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

    def new_threshold(self, local_measure, init_acc):
        """
        Responsável por calcular o novo limiar

        Args:
            local_measure: valor da acurácia do modelo treinado
            init_acc: valor da acurácia inicial
        """
        if local_measure > (init_acc + 0.01) and (
            (self.threshold - self.cr) > 0.0
        ):
            self.threshold -= self.cr
        elif (local_measure < (init_acc - 0.01)) and (
            (self.threshold + self.cr) <= 1
        ):
            self.threshold += self.cr
        else:
            pass

    def update_memory(
        self, instances: List, labels: List, weights: Optional[List] = None
    ) -> None:
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

    def storage_predict(
        self, idx, confidence, classes
    ) -> Dict[int, Dict[float, int]]:
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
            memo[i] = {}
            memo[i]["confidence"] = conf
            memo[i]["classes"] = label

        return memo

    def rule_1(self):
        """
        Regra responsável por verificar se as classes são iguais E as duas
            confianças preditas é maior que o limiar

        Returns:
            a lista correspondente pela condição
        """
        selected = []
        classes_selected = []

        for i in self.pred_x_it:
            if (
                self.dict_first[i]["confidence"] >= self.threshold
                and self.pred_x_it[i]["confidence"] >= self.threshold
                and self.dict_first[i]["classes"]
                == self.pred_x_it[i]["classes"]
            ):
                selected.append(i)
                classes_selected.append(self.dict_first[i]["classes"])

        return selected, classes_selected

    def rule_2(self):
        """
        regra responsável por verificar se as classes são iguais E uma das
        confianças preditas é maior que o limiar

        Returns:
        a lista correspondente pela condição
        """

        selected = []
        classes_selected = []

        for i in self.pred_x_it:
            if (
                self.dict_first[i]["confidence"] >= self.threshold
                or self.pred_x_it[i]["confidence"] >= self.threshold
            ) and self.dict_first[i]["classes"] == self.pred_x_it[i][
                "classes"
            ]:
                selected.append(i)
                classes_selected.append(self.dict_first[i]["classes"])

        return selected, classes_selected

    def rule_3(self):
        """
        regra responsável por verificar se as classes são diferentes E  as
        confianças preditas são maiores que o limiar

        Returns:
        a lista correspondente pela condição
        """
        selected = []

        for i in self.pred_x_it:
            if (
                self.dict_first[i]["classes"] != self.pred_x_it[i]["classes"]
                and self.dict_first[i]["confidence"] >= self.threshold
                and self.pred_x_it[i]["confidence"] >= self.threshold
            ):
                selected.append(i)

        return selected, self.remember(selected)

    def rule_4(self):
        """
        regra responsável por verificar se as classes são diferentes E uma das
        confianças preditas é maior que o limiar

        Returns:
        a lista correspondente pela condição
        """
        selected = []

        for i in self.pred_x_it:
            if self.dict_first[i]["classes"] != self.pred_x_it[i][
                "classes"
            ] and (
                self.dict_first[i]["confidence"] >= self.threshold
                or self.pred_x_it[i]["confidence"] >= self.threshold
            ):
                selected.append(i)

        return selected, self.remember(selected)

    def train_new_classifier(self, has_label, X, y):
        """
        Responsável por treinar um classificador e mensurar
            a sua acertividade

        Args:
            has_label: lista com as instâncias rotuladas
            X: instâncias
            y: rótulos

        Returns:
            Acurácia do modelo
        """

        self.transduction_ = np.copy(y)
        self.labeled_iter_ = np.full_like(y, -1)
        self.labeled_iter_[has_label] = 0
        self.init_labeled_ = has_label.copy()

        base_estimator_init = clone(self.base_estimator)
        # L0 - MODELO TREINADO E CLASSIFICADO COM L0
        base_estimator_init.fit(
            X[safe_mask(X, has_label)], self.transduction_[has_label]
        )
        # ACC EM L0 - RETORNA A EFICACIA DO MODELO
        init_acc = self.calc_local_measure(
            X[safe_mask(X, self.init_labeled_)],
            y[self.init_labeled_],
            base_estimator_init,
        )

        return init_acc

    def add_new_labeled(self, selected_full, selected, pred):
        """
        Função que retorna as intâncias rotuladas

        Args:
            selected_full: lista com os indices das instâncias originais
            selected: lista das intâncias com acc acima do limiar
            local_acc: acurácia do modelo treinado com base na lista selected
            init_acc: acurácia do modelo treinado com base na lista selected_full
            max_proba: valores de probabilidade de predição das intâncias não rotuladas
            pred: predição das instâncias não rotuladas
        """
        self.transduction_[selected_full] = pred[selected]
        self.labeled_iter_[selected_full] = self.n_iter_

        if selected_full.shape[0] > 0:
            # no changed labels
            self.new_threshold(local_acc, init_acc)
            self.termination_condition_ = "threshold_change"
        else:
            self.threshold = np.max(max_proba)

        if self.verbose:
            print(
                f"End of iteration {self.n_iter_},"
                f" added {len(selected)} new labels."
            )

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
