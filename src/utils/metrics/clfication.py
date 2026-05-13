# Python Imports

# Library Imports
import torch
from torch import Tensor

# Local Imports


class Metrics:
    def __init__(self,
                 _number_of_classes: int,
                 _predictions: Tensor,
                 _labels: Tensor):

        self.predictions: Tensor = _predictions
        self.labels: Tensor = _labels
        self.number_of_classes = _number_of_classes
        self.device = _predictions.device

        # confusion_matrix[t, p] = #voxels with true class t and predicted class p
        C = _number_of_classes
        labels_flat = _labels.reshape(-1).long()
        preds_flat = _predictions.reshape(-1).long()
        indices = labels_flat * C + preds_flat
        flat = torch.zeros(C * C, dtype=torch.int64, device=_predictions.device)
        flat.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.int64))
        self.confusion_matrix: Tensor = flat.view(C, C)

    def reportMetrics(self,
                      _metric_list,
                      _loss):
        results = {}
        results['Loss'] = _loss

        for metric in _metric_list[1:]:
            if metric == 'Accuracy':
                results[metric] = getattr(self, metric)()
            else:
                results[metric] = getattr(self, metric)(_class_id=1)

        return results

    # TP
    def TruePositive(self, _class_id) -> int:
        return self.confusion_matrix[_class_id, _class_id]

    # AP = TP + FN — all voxels whose *true* class is _class_id
    def AllPostive(self, _class_id) -> int:
        return self.confusion_matrix[_class_id, :].sum()

    # FP = #voxels predicted as _class_id whose true class is not _class_id
    def FalsePositive(self, _class_id) -> int:
        col_sum = self.confusion_matrix[:, _class_id].sum()
        return col_sum - self.confusion_matrix[_class_id, _class_id]

    # TN = #voxels neither truly _class_id nor predicted _class_id
    def TrueNegative(self, _class_id) -> int:
        total = self.confusion_matrix.sum()
        row = self.confusion_matrix[_class_id, :].sum()
        col = self.confusion_matrix[:, _class_id].sum()
        tp = self.confusion_matrix[_class_id, _class_id]
        return total - row - col + tp

    # AN = TN + FP — all voxels whose *true* class is not _class_id
    def AllNegative(self, _class_id) -> int:
        return self.confusion_matrix.sum() - self.confusion_matrix[_class_id, :].sum()

    # FN
    def FalseNegative(self, _class_id: int) -> int:
        return self.AllPostive(_class_id) - self.TruePositive(_class_id)

    # AP / (AP + AN)
    def Pervalence(self, _class_id: int) -> float:
        AP = self.AllPostive(_class_id)
        AN = self.AllNegative(_class_id)

        return AP / (AP + AN)

    # TP + TN / AP + AN
    def Accuracy(self) -> float:
        result_tensor = \
            self.confusion_matrix.diag().sum() / self.confusion_matrix.sum()
        return result_tensor

    # TPR + TNR /2
    def BalancedAccuracy(self, _class_id: int) -> float:
        return (self.TruePositiveRate(_class_id) +
                self.TrueNegativeRate(_class_id)) / 2

    # TPR(Sensitivity, Recall) = 1 - FNR
    def TruePositiveRate(self, _class_id: int) -> float:
        # TP / TP + FN = TP / AP
        TP = self.TruePositive(_class_id)
        AP = self.AllPostive(_class_id)
        return TP / AP

    # TNR(Specificity) = 1 - FPR
    def TrueNegativeRate(self, _class_id: int) -> float:
        # TN / TN + FP = TN / AN
        TN = self.TrueNegative(_class_id)
        AN = self.AllNegative(_class_id)
        return TN / AN

    # FPR(Fallout) = 1 - TNR
    def FalsePositiveRate(self, _class_id: int) -> float:
        # FP / FP + TN
        return 1 - self.TrueNegativeRate(_class_id)

    # FNR = FN / FN + TP = 1 - TPR
    def FalseNegativeRate(self, _class_id: int) -> float:
        return 1 - self.TruePositiveRate(_class_id)

    # PPV(Precision) = TP / TP + FP
    def PositivePredictiveValue(self, _class_id: int) -> float:
        TP = self.TruePositive(_class_id)
        FP = self.FalsePositive(_class_id)

        return TP / (TP + FP)

    # NPV(Miss Rate) = FN / FN + TP
    def NegativePredictiveValue(self, _class_id: int) -> float:
        TP = self.TruePositive(_class_id)
        FN = self.FalseNegative(_class_id)

        return FN / (FN + TP)

    # FDR = FP / FP + TN = 1 - PPV
    def FalseDiscoveryRate(self, _class_id: int) -> float:
        return 1 - self.PositivePredictiveValue(_class_id)

    # FOR = FN / FN + TN = 1 - NPV
    def FalseOmissionRate(self, _class_id: int) -> float:
        return 1 - self.NegativePredictiveValue(_class_id)

    # LR+ = TPR / FPR
    def PositiveLikelihoodRatio(self, _class_id):
        return self.TruePositiveRate(_class_id) / \
               self.FalsePositiveRate(_class_id)

    # LR- = FNR / TNR
    def NegativeLikelihoodRatio(self, _class_id):
        return self.FalseNegativeRate(_class_id) / \
               self.TrueNegativeRate(_class_id)

    # Dice Coefficient = 2*TP / 2*TP+FP+FN
    def Dice(self, _class_id: int) -> float:
        TP = self.TruePositive(_class_id)
        FP = self.FalsePositive(_class_id)
        FN = self.FalseNegative(_class_id)

        return 2*TP / (2*TP+FP+FN)

    # IoU  = TP / TP + FP + FN
    def JaccardIndex(self, _class_id: int) -> float:
        TP = self.TruePositive(_class_id)
        FP = self.FalsePositive(_class_id)
        FN = self.FalseNegative(_class_id)

        return TP / (TP+FP+FN)

    def FScore(self, _class_id, _betha) -> float:
        Precision = self.PositivePredictiveValue(_class_id)
        Recall = self.TruePositiveRate(_class_id)

        return (1 + _betha**2) * \
            Precision * Recall / \
            (_betha**2 * Precision + Recall)

    # MMC(Matthews correlation coefficient)
    def PhiCoefficient(self, _class_id) -> float:
        TP = self.TruePositive(_class_id)
        TN = self.TrueNegative(_class_id)
        FP = self.FalsePositive(_class_id)
        FN = self.FalseNegative(_class_id)

        TPxTN = TP * TN
        FPxFN = FP * FN

        TPpFP = TP + FP
        TPpFN = TP + FN
        TNpFP = TN + FP
        TNpFN = TN + FN

        return (TPxTN + FPxFN) / \
            (TPpFP + TPpFN + TNpFP + TNpFN)**(1/2)

    # FM = (PPV + TPR)^1/2
    def FowlkesMallowsIndex(self, _class_id) -> float:
        PPV = self.PositivePredictiveValue(_class_id)
        TPR = self.TruePositiveRate(_class_id)

        return (PPV + TPR)**(1/2)

    # DOR = LR+ / LR-
    def DiagnosticOddsRation(self, _class_id) -> float:
        return self.PositiveLikelihoodRatio(_class_id) / \
               self.NegativeLikelihoodRatio(_class_id)
