# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from deepcp.classification.predictor.base import BasePredictor
from deepcp.classification.utils import ConfCalibrator


class StandardPredictor(BasePredictor):
    def __init__(self, score_function, model):
        super().__init__(score_function, model)
    
    
    #############################
    # The confidence calibration process
    ############################
    
    def conf_calibrator(self, conf_calibration_dataloader, method, *args):
        """ Utilize conf_calibration_dataloader to optimize confidence calibrator.

        Args:
            conf_calibration_dataloader (_type_):  dataloader used to calibrate the confidence of models' output
            method:  the method of calibration method
            **args: optional parameters
        Returns:
            _type_: _description_
        """
        if type(method) == str:
            raise ValueError(f"The type of method is str.")
        logits_transformation = ConfCalibrator.registry_ConfCalibrator(method)(*args).to(self._model_device)
        self.logits_transformation = ConfCalibrator.registry_ConfOptimizer("optimze_"+method)(logits_transformation, conf_calibration_dataloader, self._model_device)
        
        
    
    #############################
    # The calibration process
    ############################
    def calibrate(self, cal_dataloader, alpha):
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for  examples in tqdm(cal_dataloader):
                tmp_x, tmp_labels = examples[0].to(self._model_device), examples[1]            
                tmp_logits = self._model(tmp_x).detach().cpu()
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
        probs = F.softmax(logits,dim=1)
        probs = probs.numpy()
        labels = labels.numpy()
        self.calculate_threshold(probs, labels, alpha)
        
        
    def calculate_threshold(self, probs, labels, alpha):
        scores = np.zeros(probs.shape[0])
        for index, (x, y) in enumerate(zip(probs, labels)):
            scores[index] = self.score_function(x, y)
        self.q_hat = np.quantile(scores, np.ceil((scores.shape[0] + 1) * (1 - alpha)) / scores.shape[0])


    #############################
    # The prediction process
    ############################
    def predict(self, x_batch):
        logits = self._model(x_batch.to(self._model_device)).detach().cpu()
        probs_batch = F.softmax(logits,dim=1).numpy()
        sets = []
        for index, probs in enumerate(probs_batch):
            sets.append(self.predict_with_probs(probs))
        return sets
    
    def predict_with_probs(self, probs):
        """ The input of score function is softmax probability.

        Args:
            probs (_type_): _description_

        Returns:
            _type_: _description_
        """
        scores = self.score_function.predict(probs)
        S = self._generate_prediction_set(scores, self.q_hat)
        return S
    
    
    #############################
    # The evaluation process
    ############################
    
    def evaluate(self, val_dataloader):
        prediction_sets = []
        labels_list = []
        with torch.no_grad():
                for  examples in tqdm(val_dataloader):
                    tmp_x, tmp_label = examples[0], examples[1]            
                    prediction_sets_batch = self.predict(tmp_x)
                    prediction_sets.extend(prediction_sets_batch)
                    labels_list.append(tmp_label)
        val_labels = torch.cat(labels_list)
        
        res_dict = {}
        res_dict["Coverage_rate"] = self._metric('coverage_rate')(prediction_sets, val_labels)
        res_dict["Average_size"] = self._metric('average_size')(prediction_sets, val_labels)
        return res_dict
        

    
    
    

