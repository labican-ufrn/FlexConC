import numpy as np
from sklearn.base import clone
from sklearn.utils import safe_mask
from abc import abstractmethod
from base_flexcons import BaseFlexCon


class FlexConC(BaseFlexCon):
    def __init__(self, base_classifier, select_model, cr=0.05, threshold=0.95, verbose=False):
        """
        FlexConC base com estratégia de seleção configurável.

        Args:
            base_classifier: Classificador base.
            select_model: Função de seleção de instâncias (FlexCon-C1(s), FlexCon-C1(v), FlexCon-C2).
            cr: Taxa de ajuste do threshold.
            threshold: Limiar de confiança para inclusão.
            verbose: Habilita mensagens de debug.
        """
        super().__init__(
            base_classifier=base_classifier,
            threshold=threshold,
            verbose=verbose
        )
        self.cr = cr
        self.threshold = threshold
        self.select_model = select_model
        self.committee_classifiers = []

    def adjust_threshold(self):
        # Ajuste do threshold baseado na acurácia local e mínima aceitável
        local_measure = self.calc_local_measure(self.X[safe_mask(self.X, np.where(self.y != -1)[0])], self.y[np.where(self.y != -1)[0]], self.classifier_)
        if local_measure > (self.init_acc + 0.01) and (self.threshold - self.cr) > 0.0:
            self.threshold -= self.cr
        elif local_measure < (self.init_acc - 0.01) and (self.threshold + self.cr) <= 1:
            self.threshold += self.cr

    def label_by_committee_sum(self, X):
        """
        Classificação por soma de probabilidades no comitê (FlexCon-C1(s)).
        """
        # Para cada classificador no comitê, calcula as probabilidades previstas para cada instância em X.
        # Isso resulta em uma lista onde cada elemento é um array de probabilidades de cada classificador.
        probabilities_list = [clf.predict_proba(X) for clf in self.committee_classifiers]

        # Calcula a média das probabilidades previstas por todos os classificadores no comitê.
        # A média é calculada para cada instância em X e para cada classe, representando a soma ponderada das opiniões do comitê.
        probabilities = np.mean(probabilities_list, axis=0)

        # Retorna o índice da classe com maior probabilidade para cada instância.
        # Isso corresponde aos rótulos finais baseados na soma das probabilidades.
        return probabilities.argmax(axis=1)

    def label_by_committee_vote(self, X):
        """
        Classificação por voto majoritário no comitê (FlexCon-C1(v)).
        """
        # Para cada classificador no comitê, realiza a predição dos rótulos para as instâncias em X.
        # Isso resulta em uma matriz onde cada linha representa as predições de um classificador.
        votes = np.array([clf.predict(X) for clf in self.committee_classifiers])

        # Inicializa uma lista para armazenar os rótulos majoritários para cada instância.
        majority_labels = []

        # Para cada coluna (instância), conta a frequência dos rótulos previstos e seleciona o mais frequente.
        for i in range(votes.shape[1]):
            # Conta a frequência dos rótulos na coluna.
            label_counts = np.bincount(votes[:, i])
            # Encontra o rótulo mais frequente.
            majority_label = np.argmax(label_counts)
            # Adiciona o rótulo à lista de resultados.
            majority_labels.append(majority_label)

        # Converte a lista de rótulos majoritários para um array e retorna.
        return np.array(majority_labels)

    def label_by_previous_iteration(self, X):
        """
        Classificação pela iteração anterior (FlexCon-C2).
        """
        return self.classifier_.predict(X)

    def select_instances_by_rules(self):
        """
        Seleção de instâncias usando a estratégia de seleção configurada.
        """
        return self.select_model(self)

def flexconC1s(self, instance):
        """
        Estratégia de seleção para FlexCon-C1(s).
        """
        # Adiciona o classificador atual ao comitê
        self.committee_classifiers.append(clone(self.classifier_))

        # Obtém os rótulos previstos usando a soma das probabilidades no comitê
        labels_sum = self.label_by_committee_sum(self.pred_x_it.keys())

        # Extrai as confianças das predições para cada instância
        confidences = [self.pred_x_it[i]['confidence'] for i in self.pred_x_it.keys()]

        # Seleciona instâncias cuja confiança seja maior ou igual ao threshold
        selected = [i for i, conf in zip(self.pred_x_it.keys(), confidences) if conf >= self.threshold]

        # Retorna as instâncias selecionadas e seus rótulos baseados na soma
        return selected, labels_sum[selected]

def flexconC1v(self, instance):
    """
    Estratégia de seleção para FlexCon-C1(v).
    """
    # Adiciona o classificador atual ao comitê
    self.committee_classifiers.append(clone(self.classifier_))

    # Obtém os rótulos previstos usando o voto majoritário no comitê
    labels_vote = self.label_by_committee_vote(self.pred_x_it.keys())

    # Extrai as confianças das predições para cada instância
    confidences = [self.pred_x_it[i]['confidence'] for i in self.pred_x_it.keys()]

    # Seleciona instâncias cuja confiança seja maior ou igual ao threshold
    selected = [i for i, conf in zip(self.pred_x_it.keys(), confidences) if conf >= self.threshold]

    # Retorna as instâncias selecionadas e seus rótulos baseados no voto
    return selected, labels_vote[selected]

def flexconC2(self, instance):
    """
    Estratégia de seleção para FlexCon-C2.
    """
    # Obtém as confianças das predições para cada instância
    confidences = [self.pred_x_it[i]['confidence'] for i in self.pred_x_it.keys()]

    # Obtém os rótulos previstos usando o classificador da iteração anterior
    labels_previous = self.label_by_previous_iteration(self.pred_x_it.keys())

    # Seleciona instâncias cuja confiança seja maior ou igual ao threshold
    selected = [i for i, conf in zip(self.pred_x_it.keys(), confidences) if conf >= self.threshold]

    # Retorna as instâncias selecionadas e seus rótulos baseados na iteração anterior
    return selected, labels_previous[selected]