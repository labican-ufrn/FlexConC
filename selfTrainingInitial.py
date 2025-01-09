import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array


class SelfTrainingClassifier:
    def __init__(
        self,
        base_classifier,
        criterion="threshold",
        threshold=0.75,
        k_best=10,
        max_iter=10,
    ):
        self.base_classifier = base_classifier
        self.criterion = criterion
        self.threshold = threshold
        self.k_best = k_best
        self.max_iter = max_iter
        self.classifier_ = None

    def fit(self, X, y):
        X = check_array(X)
        self.classifier_ = clone(self.base_classifier)

        # Inicializa com dados rotulados
        labeled_mask = ~np.isnan(y)
        X_labeled, y_labeled = X[labeled_mask], y[labeled_mask]

        # Treina inicialmente com dados rotulados
        self.classifier_.fit(X_labeled, y_labeled)

        # Realimentação
        for iteration in range(self.max_iter):
            try:
                # Previsões e probabilidades para dados não rotulados
                unlabeled_mask = np.isnan(y)
                if not np.any(unlabeled_mask):
                    break  # Para se não houver dados não rotulados
                X_unlabeled = X[unlabeled_mask]
                probs = self.classifier_.predict_proba(X_unlabeled)

                # Seleciona instâncias com alta confiança
                max_probs = probs.max(axis=1)
                if self.criterion == "threshold":
                    confident_samples = max_probs >= self.threshold
                elif self.criterion == "k_best":
                    n_to_select = min(self.k_best, max_probs.shape[0])
                    if n_to_select == 0:
                        break
                    confident_samples_indices = np.argpartition(-max_probs, n_to_select)[:n_to_select]
                    confident_samples = np.zeros_like(max_probs, dtype=bool)
                    confident_samples[confident_samples_indices] = True
                else:
                    raise ValueError("Invalid criterion. Choose either 'threshold' or 'k_best'.")

                if not np.any(confident_samples):
                    break  # Para se não houver instâncias confiantes

                # Atualiza rótulos dos dados não rotulados com alta confiança
                new_y = self.classifier_.predict(X_unlabeled[confident_samples])
                y[unlabeled_mask][confident_samples] = new_y

                # Re-treina o classificador com o novo conjunto de dados rotulados
                X_labeled, y_labeled = X[~np.isnan(y)], y[~np.isnan(y)]
                self.classifier_.fit(X_labeled, y_labeled)

            except NotFittedError:
                print("O classificador base não foi treinado corretamente.")
                break

    def predict(self, X):
        if not self.classifier_:
            raise NotFittedError("O classificador ainda não foi treinado.")
        return self.classifier_.predict(X)

    def predict_proba(self, X):
        if not self.classifier_:
            raise NotFittedError("O classificador ainda não foi treinado.")
        return self.classifier_.predict_proba(X)
