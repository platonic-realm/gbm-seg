"""
Author: Arash Fatehi
Date:   06.02.2022
Links: https://en.wikipedia.org/wiki/Confusion_matrix
"""

# Python Imports

# Library Imports
import torch
from torch import Tensor

# Local Imports


class Metrics():
    def __init__(self,
                 _number_of_classes: int,
                 _predictions: Tensor,
                 _labels: Tensor):

        self.predictions: Tensor = _predictions
        self.labels: Tensor = _labels
        self.number_of_classes = _number_of_classes
        self.device = _predictions.device

        # Calculating the confusion matrix
        self.confusion_matrix: Tensor = torch.zeros(_number_of_classes,
                                                    _number_of_classes,
                                                    dtype=torch.int32,
                                                    device=_predictions.device,
                                                    requires_grad=False)
        for i in range(self.number_of_classes):
            self.confusion_matrix[:, i] = \
                    self.calcualte_confusion_matrix_for_class_id(i)

    def calcualte_confusion_matrix_for_class_id(self,
                                                _class_id: int) -> Tensor:

        assert _class_id < self.number_of_classes, \
                "Out of range index, class ids "\
                "should be smaller than number of classes"

        result = torch.zeros(self.number_of_classes,
                             dtype=torch.int32,
                             device=self.device)

        # Copy the predictions tensor as next operation are destructive
        # And also detach it from the computation graph to save computation
        local_predictions = self.predictions.clone().detach()

        # To calulate number of correct and wrong predictions of
        # _truth_class_id, we need to first create a mask consists of
        # indexs in the labels tensor with the value of _class_id.
        mask = (self.labels == _class_id).float()

        # Having a class_id == 0 will cause problem when we count
        # number of predictions for each class, so change it here
        local_predictions[local_predictions == 0] = self.number_of_classes

        # Using mask to filter the predictions based on
        # the ground truth for the class_id
        local_predictions = local_predictions * mask

        for i in range(self.number_of_classes):
            # For the sake of readablity, I have just used
            # an if instead of a branchless approch
            # Plus: Python is slow anyway
            if i == 0:
                result[i] = \
                    (local_predictions == self.number_of_classes).int().sum()
            else:
                result[i] = \
                    (local_predictions == i).int().sum()

        return result

    # TP
    def TruePositive(self, _class_id) -> int:
        result_tensor = self.confusion_matrix[_class_id, _class_id]
        return result_tensor

    # AP = TP + FN
    def AllPostive(self, _class_id) -> int:
        result_tensor = self.confusion_matrix[:, _class_id].sum()
        return result_tensor

    def FalsePositive(self, _class_id) -> int:
        FP = torch.zeros(1, device=self.device)
        for i in range(self.number_of_classes):
            for j in range(self.number_of_classes):
                if _class_id == i and _class_id != j:
                    FP += self.confusion_matrix[i, j]
        return FP

    # TN
    def TrueNegative(self, _class_id) -> int:
        TN = torch.zeros(1, device=self.device)
        for i in range(self.number_of_classes):
            for j in range(self.number_of_classes):
                if _class_id not in (i, j):
                    TN += self.confusion_matrix[i, j]
        return TN

    # AN = TN + FP
    def AllNegative(self, _class_id) -> int:
        TN = torch.zeros(1, device=self.device)
        FP = torch.zeros(1, device=self.device)
        for i in range(self.number_of_classes):
            for j in range(self.number_of_classes):
                if _class_id not in (i, j):
                    TN += self.confusion_matrix[i, j]
                elif _class_id == i and _class_id != j:
                    FP += self.confusion_matrix[i, j]
        return TN + FP

    # FN
    def FalseNegative(self, _class_id: int) -> int:
        return self.AllPostive(_class_id) - self.TruePositive(_class_id)

    # AP / AP + AN
    def Pervalence(self, _class_id: int) -> float:
        AP = self.AllPostive(_class_id)
        AN = self.AllNegative(_class_id)

        return AP / AP + AN

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
