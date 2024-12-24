from base_flexcons import BaseFlexCon
import numpy as np


class FlexConG(BaseFlexCon):
    def __init__(self, base_classifier, cr=0.05, threshold=0.95, verbose=False):
        super().__init__(base_classifier=base_classifier, threshold=threshold, verbose=verbose)
        self.cr = cr

    def fit(self, X, y):
        # Inicialização do classificador e acurácia inicial
        labeled_indices = np.where(y != -1)[0]
        unlabeled_indices = np.where(y == -1)[0]

        init_acc = self.train_new_classifier(labeled_indices, X, y)

        # Iteração até que não haja mais instâncias para rotular ou limite de iterações
        for self.n_iter_ in range(self.max_iter):

            # Verifica se ainda há instâncias não rotuladas
            if len(unlabeled_indices) == 0:
                break

            self.pred_x_it = self.storage_predict(unlabeled_indices, self.classifier_.predict_proba(X[unlabeled_indices]).max(axis=1), self.classifier_.predict(X[unlabeled_indices]))
            selected_indices, predictions = self.select_instances_by_rules()

            if len(selected_indices) == 0:
                break

            # Atualizar o conjunto de instâncias rotuladas
            self.add_new_labeled(selected_indices, selected_indices, predictions)

            # Ajustar o threshold
            if (self.threshold - self.cr) > 0.0:
                self.threshold -= self.cr
            else:
                break

            # Treinar o classificador com os novos dados rotulados
            init_acc = self.train_new_classifier(labeled_indices, X, y)

        return self
