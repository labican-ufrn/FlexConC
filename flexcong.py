from mlabican import BaseFlexCon


class FlexConG(BaseFlexCon):
    """
    Classe que implementa o método FlexConG

    Args:
        BaseFlexCon (Object)
    """
    def __init__(
        self,
        base_classifier,
        cr=0.05,
        threshold=0.95,
        max_iter=10,
        verbose=False
    ):
        """
        Inicializa a classe FlexConG

        Args:
            base_classifier (Object): Classe base FlexCon.
            cr (float): Taxa de variação do threshold. Defaults to 0.05.
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
        self.cr = cr

    def adjust_threshold(self):
        """
        Ajusta o threshold dinamicamente com a lógica do FlexConG.
        """
        if (self.threshold - self.cr) > 0.0:
            self.threshold -= self.cr
