import logging
import math
from collections import OrderedDict
import numpy as np
from scipy import spatial
from .defense_base import BaseDefenseMethod
from typing import List, Tuple, Dict, Any
from ..common.utils import (
    compute_euclidean_distance,
    compute_middle_point,
    compute_krum_score, compute_gaussian_distribution
)
import torch
import fedml


class ThreeSigmaDefense(BaseDefenseMethod):
    def __init__(self, config):
        self.average = None
        self.upper_bound = 0
        self.malicious_client_idxs = []
        # OutlierDetectionDefense will set this list; when it is empty, kick out detected malicious models directly
        self.potential_malicious_client_idxs = None
        self.device = fedml.device.get_device(config)
        self.lambda_value = 3  # default: 0.5
        # if hasattr(config, "bound_param") and isinstance(config.bound_param, float):
        #     self.bound_param = config.bound_param
        # else:
        #     self.bound_param = 1

    ###################### version 3: re-compute gaussian distribution each round
    def defend_before_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
    ):
        importance_feature_list = self._get_importance_feature(raw_client_grad_list)
        if self.average is None:
            self.average = self.compute_avg_with_krum(importance_feature_list)
        client_scores = self.compute_l2_scores(importance_feature_list)
        mu, sigma = compute_gaussian_distribution(client_scores)
        # self.upper_bound = mu + self.bound_param * sigma
        # self.upper_bound0_1 = mu + 0.1 * sigma
        # self.upper_bound0_3 = mu + 0.3 * sigma
        # self.upper_bound0_5 = mu + 0.5 * sigma
        self.upper_bound = mu + self.lambda_value * sigma
        # self.upper_bound1_5 = mu + 1.5 * sigma
        # self.upper_bound2 = mu + 2 * sigma
        # self.upper_bound2_5 = mu + 2.5 * sigma
        # self.upper_bound3 = mu + 3 * sigma
        # print(f"client socres = {client_scores}")
        # print(f"mu = {mu}, sigma = {sigma}, upperbound = {self.upper_bound}")
        new_client_models, _ = self.kick_out_poisoned_local_models(
            client_scores, raw_client_grad_list
        )
        importance_feature_list = self._get_importance_feature(new_client_models)
        self.average = self.compute_an_average_feature(importance_feature_list)
        return new_client_models

    def compute_an_average_feature(self, importance_feature_list):
        alphas = [1 / len(importance_feature_list)] * len(importance_feature_list)
        return compute_middle_point(alphas, importance_feature_list)

    ##################### version 2: remove poisoned model scores in score list
    # def defend_before_aggregation(
    #     self,
    #     raw_client_grad_list: List[Tuple[float, OrderedDict]],
    #     extra_auxiliary_info: Any = None,
    # ):
    #     if self.median is None:
    #         self.median = self.compute_median_with_krum(raw_client_grad_list)
    #     client_scores = self.compute_scores(raw_client_grad_list)
    #     print(f"client scores = {client_scores}")
    #     if self.iteration_num < self.pretraining_round_number:
    #         mu, sigma = compute_gaussian_distribution(self.score_list, client_scores)
    #         self.upper_bound = mu + self.bound_param * sigma
    #         print(f"mu = {mu}, sigma = {sigma}, upperbound = {self.upper_bound}")
    #         new_client_models, client_scores = self.kick_out_poisoned_local_models(client_scores, raw_client_grad_list)
    #         print(f"new scores after kicking out = {client_scores}")
    #         self.score_list.extend(list(client_scores))
    #         mu, sigma = compute_gaussian_distribution(self.score_list, [])
    #         self.upper_bound = mu + self.bound_param * sigma
    #         print(f"mu = {mu}, sigma = {sigma}, upperbound = {self.upper_bound}")
    #     else:
    #         new_client_models, _ = self.kick_out_poisoned_local_models(client_scores, raw_client_grad_list)
    #     self.iteration_num += 1
    #     return new_client_models

    ###################### version 1: do not remove poisoned model scores in score list
    # def defend_before_aggregation(
    #     self,
    #     raw_client_grad_list: List[Tuple[float, OrderedDict]],
    #     extra_auxiliary_info: Any = None,
    # ):
    #     if self.median is None:
    #         self.median = self.compute_median_with_krum(raw_client_grad_list)
    #     client_scores = self.compute_scores(raw_client_grad_list)
    #     print(f"client scores = {client_scores}")
    #
    #     if self.iteration_num < self.pretraining_round_number:
    #         self.score_list.extend(list(client_scores))
    #         self.mu, self.sigma = compute_gaussian_distribution(self.score_list)
    #         self.upper_bound = self.mu + self.bound_param * self.sigma
    #         self.iteration_num += 1
    #
    #     for i in range(len(client_scores) - 1, -1, -1):
    #         if client_scores[i] > self.upper_bound:
    #      # we do not remove the score in self.score_list to avoid mis-deleting due to severe non-iid among clients
    #             raw_client_grad_list.pop(i)
    #             print(f"pop -- i = {i}")
    #     return raw_client_grad_list

    def kick_out_poisoned_local_models(self, client_scores, raw_client_grad_list):
        # print(f"upper bound = {self.upper_bound}")
        # traverse the score list in a reversed order
        # malicious_client_idxs_0_5dev = []
        # malicious_client_idxs_0_1dev = []
        # malicious_client_idxs_0_3dev = []
        self.malicious_client_idxs = []
        # malicious_client_idxs_1_5dev = []
        # # logging.info(f"!!!!!!!!!!!!!!!!!!!!potential_malicious_client_idxs = {self.potential_malicious_client_idxs}")
        # # self.log_file.write(f"potential_malicious_client_idxs = {self.potential_malicious_client_idxs}")
        # malicious_client_idxs_2dev = []
        # malicious_client_idxs_2_5dev = []
        # malicious_client_idxs_3dev = []
        for i in range(len(client_scores) - 1, -1, -1):
            # if client_scores[i] > self.upper_bound0_1:
            #     malicious_client_idxs_0_1dev.append(i)
            # if client_scores[i] > self.upper_bound0_3:
            #     malicious_client_idxs_0_3dev.append(i)
            # if client_scores[i] > self.upper_bound0_5:
            #     malicious_client_idxs_0_5dev.append(i)
            # if client_scores[i] > self.upper_bound1_5:
            #     malicious_client_idxs_1_5dev.append(i)
            # if client_scores[i] > self.upper_bound2:
            #     malicious_client_idxs_2dev.append(i)
            # if client_scores[i] > self.upper_bound2_5:
            #     malicious_client_idxs_2_5dev.append(i)
            # if client_scores[i] > self.upper_bound3:
            #     malicious_client_idxs_3dev.append(i)
            if client_scores[i] > self.upper_bound:
                logging.info(f"second phase: scores that exeed bound: {i}")
                if self.potential_malicious_client_idxs is None or i in self.potential_malicious_client_idxs:
                    raw_client_grad_list.pop(i)
                    self.malicious_client_idxs.append(i)
                    # logging.info(f"kick out -- {i}")
        # logging.info(f"!!!!!!!!!!!!!!!!!!!!detected malicious0.1dev: {malicious_client_idxs_0_1dev}")
        # logging.info(f"!!!!!!!!!!!!!!!!!!!!detected malicious0.3dev: {malicious_client_idxs_0_3dev}")
        # logging.info(f"!!!!!!!!!!!!!!!!!!!!detected malicious0.5dev: {malicious_client_idxs_0_5dev}")
        logging.info(f"!!!!!!!!!!!!!!!!!!!!second phase: detected malicious: {self.malicious_client_idxs}")
        # logging.info(f"!!!!!!!!!!!!!!!!!!!!detected malicious1.5dev: {malicious_client_idxs_1_5dev}")
        # logging.info(f"!!!!!!!!!!!!!!!!!!!!detected malicious2dev: {malicious_client_idxs_2dev}")
        # logging.info(f"!!!!!!!!!!!!!!!!!!!!detected malicious2.5dev: {malicious_client_idxs_2_5dev}")
        # logging.info(f"!!!!!!!!!!!!!!!!!!!!detected malicious3dev: {malicious_client_idxs_3dev}")
        # self.log_file.write(f"detected malicious: {self.malicious_client_idxs}\n\n")
        return raw_client_grad_list, client_scores

    def get_malicious_client_idxs(self):
        return self.malicious_client_idxs

    def set_potential_malicious_clients(self, potential_malicious_client_idxs):
        self.potential_malicious_client_idxs = potential_malicious_client_idxs

    def compute_avg_with_krum(self, importance_feature_list):
        krum_scores = compute_krum_score(
            importance_feature_list,
            client_num_after_trim=math.ceil(len(importance_feature_list) / 2) - 1,
        )
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: math.ceil(len(importance_feature_list) / 2) - 1]
        honest_importance_feature_list = [
            importance_feature_list[i] for i in score_index
        ]
        return self.compute_an_average_feature(honest_importance_feature_list)

    def compute_l2_scores(self, importance_feature_list):
        scores = []
        for feature in importance_feature_list:
            score = compute_euclidean_distance(torch.Tensor(feature), self.average)
            scores.append(score)
        return scores

    # def compute_client_cosine_scores(self, raw_client_grad_list):
    #     importance_feature_list = get_importance_feature(raw_client_grad_list)
    #     cosine_scores = []
    #     num_client = len(importance_feature_list)
    #     for i in range(0, num_client):
    #         dists = []
    #         for j in range(0, num_client):
    #             if i != j:
    #                 dists.append(
    #                     1
    #                     - spatial.distance.cosine(
    #                         importance_feature_list[i], importance_feature_list[j]
    #                     )
    #                 )
    #         cosine_scores.append(sum(dists) / len(dists))
    #     return cosine_scores

    def _get_importance_feature(self, raw_client_grad_list):
        # print(f"raw_client_grad_list = {raw_client_grad_list}")
        # Foolsgold uses the last layer's gradient/weights as the importance feature.
        ret_feature_vector_list = []
        for idx in range(len(raw_client_grad_list)):
            raw_grad = raw_client_grad_list[idx]
            (p, grads) = raw_grad

            # Get last key-value tuple
            (weight_name, importance_feature) = list(grads.items())[-2]
            # print(importance_feature)
            feature_len = np.array(
                importance_feature.cpu().data.detach().numpy().shape
            ).prod()
            feature_vector = importance_feature.cpu().data.detach().view(feature_len)
            # feature_vector = np.reshape(
            #     importance_feature.cpu().data.detach().numpy(), feature_len
            # )
            ret_feature_vector_list.append(feature_vector)
        return ret_feature_vector_list
