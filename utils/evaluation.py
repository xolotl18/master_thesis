import numpy as np
from statistics import mean

class Evaluate():
    def __init__(self, predictions):
        self.predictions = predictions
        
    def dice(self, input, target):
      #this is a metric computed from numpy arrays
      #the dice coefficient or F1 score is given by the intersection 
      #of the prediction mask and the ground truth divided by
      #the sum of the pixels belonging to the prediction 
      #and the pixels belonging to the ground truth

      l_input = input.astype(bool)
      l_target = target.astype(bool)
      intersection = np.logical_and(l_input, l_target)
      union = np.logical_or(l_input, l_target)
      dice = 2*(np.sum(intersection))/(np.sum(union)+np.sum(intersection))
      return dice
        
    def jaccard(self, input, target):
      #this is a metric computed from numpy arrays
      #the jaccard index or interserction over union is given by
      #the intersection of the prediction mask and the ground truth
      #divided by the union of the prediction mask and the gt
      #the union is the sum of the pixels minus their intersection

      l_input = input.astype(bool)
      l_target = target.astype(bool)
      intersection = np.logical_and(l_input, l_target)
      union = np.logical_or(l_input, l_target)
      iou = np.sum(intersection)/np.sum(union)
      return iou

    def get_IoU(self):
        #returns the mean iou over a series of predictions
        IoUs = []
        for pred, target, _, _ in self.predictions:
            iou = self.jaccard(pred, target)
            IoUs.append(iou)
            
        IoU_score = mean(IoUs)
        return IoU_score
    
    def get_dice(self):
        #returns the mean f1 score over a series of predictions
        F1s = []
        for pred, target, _, _ in self.predictions:
            f1 = self.dice(pred, target)
            F1s.append(f1)
            
        F1_score = mean(F1s)
        return F1_score

    def get_metrics(self):
        #returns both iou and f1 score in a dictionary
        IoUs = []
        F1s = []
        for pred, target, _, _ in self.predictions:
            iou = self.jaccard(pred, target)
            f1 = self.dice(pred, target)
            IoUs.append(iou)
            F1s.append(f1)
            
        IoU_score = mean(IoUs)
        F1_score = mean(F1s)
        
        metrics = {
            "IoU" : IoU_score,
            "Dice" : F1_score
        }
        return metrics
        