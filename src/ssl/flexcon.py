from abc import abstractmethod

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

    def __init__(
        self,
        base_estimator,
        cr: float = 0.05,
        threshold: float = 0.95,
        verbose: bool = False
    ):
        super().__init__(
            base_estimator=base_estimator,
            threshold=threshold,
            max_iter=100
        )
        self.cr: float = cr
        self.threshold: float = threshold
        self.verbose: bool = verbose
        self.old_selected: list = []
        self.dict_first: dict = {}
        self.init_labeled_: list = []
        self.labeled_iter_: list = []
        self.transduction_: list = []
        self.classes_: list = []
        self.termination_condition_ = ""
        self.pred_x_it: dict = {}
        self.cl_memory: list = []
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_select_ = clone(self.base_estimator)

    def __str__(self) -> str:
        msg = super().__str__()
        msg += (
            f"Classificador {self.base_estimator}\n"
            f"Outros Parâmetro:"
            f" CR: {self.cr}\t Threshold: {self.threshold}"
            f" Máximo IT: {self.max_iter}"
        )

        return msg

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError('Implement me')

    def validate(self):
        """
        Validate the fitted estimator since `predict_proba` can be
        delegated to an underlying "final" fitted estimator as
        generally done in meta-estimator or pipeline.
        """
        try:
            if not hasattr(self.base_estimator, "predict_proba"):
                msg = "base_estimator ({}) should implement predict_proba!"
                raise ValueError(
                    msg.format(type(self.base_estimator).__name__)
                )
        except ValueError:
            return False

        return True

    def calc_local_measure(
        self,
        X: list,
        y_true: list,
        classifier
    ) -> float:
        """
        Calcula o valor da acurácia do modelo.

        Args:
            - X: instâncias.
            - y_true: classes.
            - classifier: modelo.

        Returns:
            Retorna a acurácia do modelo.
        """
        y_pred = classifier.predict(X)

        return accuracy_score(y_true, y_pred)

    def new_threshold(
        self,
        local_measure: float,
        init_acc: float
    ) -> None:
        """
        Responsável por calcular o novo limiar.

        Args:
            - local_measure: valor da acurácia do modelo treinado.
            - init_acc: valor da acurácia inicial.
        """

        if local_measure > (init_acc + 0.01) and (
            (self.threshold - self.cr) > 0.0
        ):
            self.threshold -= self.cr
        elif (local_measure < (init_acc - 0.01)) and (
            (self.threshold + self.cr) <= 1
        ):
            self.threshold += self.cr

    def update_memory(
        self,
        instances: list,
        labels: list,
        weights: list | None = None,
    ) -> None:
        """
        Atualiza a matriz de instâncias rotuladas.

        Args:
            - instances: instâncias.
            - labels: rotulos.
            - weights: Pesos de cada classe.
        """

        if not weights:
            weights = [1 for _ in range(len(instances))]

        for instance, label, weight in zip(instances, labels, weights):
            self.cl_memory[instance][label] += weight

    def remember(self, X: list) -> list:
        """
        Responsável por armazenar como está as instâncias dado um
        momento no código.

        Args:
            - X: lista com as instâncias.

        Returns:
            A lista memorizada em um dado momento.
        """

        return [np.argmax(self.cl_memory[x]) for x in X]

    def storage_predict(
        self,
        idx,
        confidence,
        classes,
    ) -> dict[int, dict[float, int]]:
        """
        Responsável por armazenar o dicionário de dados da matriz.

        Args:
            - idx: indices de cada instância.
            - confidence: taxa de confiança para a classe destinada.
            - classes: indices das classes.

        Returns:
            Retorna o dicionário com as classes das instâncias não
            rotuladas.

            .. code-block: python
            memo = {
                <idx>: {
                    'confidence': 0.9,
                    'classes': 3
                }
                # ... outras instancias
            }

        """
        memo = {}

        for instance, conf, label in zip(idx, confidence, classes):
            memo[instance] = {}
            memo[instance]["confidence"] = conf
            memo[instance]["classes"] = label

        return memo

    def rule_1(self):
        """
        Regra responsável por verificar se as classes são iguais E as duas
            confianças preditas é maior que o limiar

        Returns:
            - A lista correspondente pela condição;
            - A classe de cada elemento da lista.
        """
        selected = []
        classes_selected = []

        for instance, values in self.pred_x_it.items():
            if (
                self.dict_first[instance]["confidence"] >= self.threshold
                and values["confidence"] >= self.threshold
                and self.dict_first[instance]["classes"]
                == values["classes"]
            ):
                selected.append(instance)
                classes_selected.append(self.dict_first[instance]["classes"])

        return selected, classes_selected

    def rule_2(self):
        """
        regra responsável por verificar se as classes são iguais E uma das
        confianças preditas é maior que o limiar

        Returns:
            - A lista correspondente pela condição;
            - A classe de cada elemento da lista.
        """

        selected = []
        classes_selected = []

        for instance, values in self.pred_x_it.items():
            if (
                self.dict_first[instance]["confidence"] >= self.threshold
                or values["confidence"] >= self.threshold
            ) and self.dict_first[instance]["classes"] == values["classes"]:
                selected.append(instance)
                classes_selected.append(self.dict_first[instance]["classes"])

        return selected, classes_selected

    def rule_3(self) -> tuple:
        """
        regra responsável por verificar se as classes são diferentes E  as
        confianças preditas são maiores que o limiar

        Returns:
            - A lista correspondente pela condição;
            - A classe de cada elemento da lista.
        """
        selected = []

        for instance, values in self.pred_x_it.items():
            if (
                self.dict_first[instance]["classes"] != values["classes"]
                and self.dict_first[instance]["confidence"] >= self.threshold
                and values["confidence"] >= self.threshold
            ):
                selected.append(instance)

        return selected, self.remember(selected)

    def rule_4(self) -> tuple:
        """
        regra responsável por verificar se as classes são diferentes E uma das
        confianças preditas é maior que o limiar

        Returns:
            - A lista correspondente pela condição;
            - A classe de cada elemento da lista.
        """
        selected = []

        for instance, values in self.pred_x_it.items():
            if (
                self.dict_first[instance]["classes"] != values["classes"]
                and (
                    self.dict_first[instance]["confidence"] >= self.threshold
                    or values["confidence"] >= self.threshold
                )
            ):
                selected.append(instance)

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

    def add_new_labeled(
        self,
        selected_full: list,
        selected: list,
        local_acc: float,
        init_acc: float,
        max_proba: list[float],
        pred: list,
    ):
        """
        Função que retorna as intâncias rotuladas

        Args:
            - selected_full: lista com os indices das instâncias
                originais.
            - selected: lista das intâncias com acc acima do limiar.
            - local_acc: acurácia do modelo treinado com base na lista
                selected.
            - init_acc: acurácia do modelo treinado com base na lista
                selected_full.
            - max_proba: valores de probabilidade de predição das
                intâncias não rotuladas.
            - pred: predição das instâncias não rotuladas.
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

    def select_instances_by_rules(self) -> tuple:
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
