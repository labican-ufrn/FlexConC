from base_flexcons import BaseFlexCon


class FlexConG(BaseFlexCon):
    def __init__(self, base_classifier, cr=0.05, threshold=0.95, max_iter=10, verbose=False):
        super().__init__(base_classifier=base_classifier, threshold=threshold, max_iter=max_iter, verbose=verbose)
        self.cr = cr

    def adjust_threshold(self):
        if (self.threshold - self.cr) > 0.0:
            self.threshold -= self.cr