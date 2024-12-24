from base_flexcons import BaseFlexCon
import numpy as np


class FlexCon(BaseFlexCon):
    def __init__(self, base_classifier, threshold=0.95, verbose=False):
        super().__init__(base_classifier=base_classifier, threshold=threshold, verbose=verbose)

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
            self.pred_x_it = self.storage_predict(unlabeled_indices, self.classifier_.predict_proba(X[unlabeled_indices]).max(axis=1), self.classifier_.predict(X[unlabeled_indices]))
            selected_indices, predictions = self.select_instances_by_rules()

            if len(selected_indices) == 0:
                break

            # Atualizar o conjunto de instâncias rotuladas
            self.add_new_labeled(selected_indices, selected_indices, predictions)

            # Ajuste do threshold com base em acurácia, confiança e taxa de instâncias rotuladas
            labeled_count = len(np.where(y != -1)[0])
            unlabeled_count = len(np.where(y == -1)[0])
            self.threshold = (self.threshold + init_acc + (labeled_count / unlabeled_count)) / 3

            # Re-treinar o classificador com o novo conjunto de dados rotulados
            init_acc = self.train_new_classifier(labeled_indices, X, y)

        return self
