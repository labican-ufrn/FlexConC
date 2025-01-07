from base_flexcons import BaseFlexCon
import numpy as np


class FlexCon(BaseFlexCon):
    def __init__(self, base_classifier, threshold=0.95, max_iter=10, verbose=False):
        super().__init__(base_classifier=base_classifier, threshold=threshold, max_iter=max_iter, verbose=verbose)

    def adjust_threshold(self, local_measure):
        labeled_count = len(np.where(self.transduction_ != -1)[0])
        unlabeled_count = len(np.where(self.transduction_ == -1)[0])
        self.threshold = (self.threshold + local_measure + (labeled_count / unlabeled_count)) / 3