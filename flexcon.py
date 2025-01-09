from mlabican import BaseFlexCon
from numpy import where


class FlexCon(BaseFlexCon):
    """
    Classe que implementa o método FlexCon

    Args:
        BaseFlexCon (Object)
    """
    def __init__(
        self,
        base_classifier,
        threshold=0.95,
        max_iter=10,
        verbose=False
    ):
        """
        Inicializa a classe FlexCon

        Args:
            base_classifier (Object): Classe base FlexCon.
            threshold (float): Valor inicial do threshold. Defaults to 0.95.
            max_iter (int): Número máximo de iterações. Defaults to 10.
            verbose (bool): Controle de exibição de mensagem. Defaults to False
        """
        super().__init__(
            base_classifier=base_classifier,
            threshold=threshold,
            max_iter=max_iter,
            verbose=verbose
        )

    def adjust_threshold(self, local_measure):
        """
        Ajusta o threshold dinamicamente com a lógica do FlexCon.
        """
        labeled_count = len(where(self.transduction_ != -1)[0])
        unlabeled_count = len(where(self.transduction_ == -1)[0])
        self.threshold = (self.threshold + local_measure +
            (labeled_count / unlabeled_count)) / 3
