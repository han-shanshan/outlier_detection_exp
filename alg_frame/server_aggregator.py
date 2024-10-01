import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple
from FederatedLearning.core import Context
from ..contribution.contribution_assessor_manager import ContributionAssessorManager
from ..dp.FederatedLearning_differential_privacy import FederatedLearningDifferentialPrivacy
from ..security.attacker import FederatedLearningAttacker
from ..security.defender import FederatedLearningDefender
from FederatedLearning.ml.aggregator.agg_operator import FederatedLearningAggOperator
from ..fhe.fhe_agg import FederatedLearningFHE


class ServerAggregator(ABC):
    """Abstract base class for federated learning trainer."""

    def __init__(self, model, args):
        self.model = model
        self.id = 0
        self.args = args
        FederatedLearningAttacker.get_instance().init(args)
        FederatedLearningDefender.get_instance().init(args)
        FederatedLearningDifferentialPrivacy.get_instance().init(args)
        FederatedLearningFHE.get_instance().init(args)
        self.contribution_assessor_mgr = ContributionAssessorManager(args)
        self.final_contribution_assigment_dict = dict()

        self.eval_data = None

    def is_main_process(self):
        return True

    def set_id(self, aggregator_id):
        self.id = aggregator_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    def on_before_aggregation(
            self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]
    ):
        if FederatedLearningFHE.get_instance().is_fhe_enabled():
            logging.info(" ---- loading encrypted models ----")
            enc_raw_client_model_or_grad_list = raw_client_model_or_grad_list
            client_idxs = [i for i in range(len(raw_client_model_or_grad_list))]
            return enc_raw_client_model_or_grad_list, client_idxs
        else:
            if FederatedLearningDifferentialPrivacy.get_instance().is_global_dp_enabled() and FederatedLearningDifferentialPrivacy.get_instance().is_clipping():
                raw_client_model_or_grad_list = FederatedLearningDifferentialPrivacy.get_instance().global_clip(raw_client_model_or_grad_list)
            if FederatedLearningAttacker.get_instance().is_data_reconstruction_attack():
                FederatedLearningAttacker.get_instance().reconstruct_data(
                    raw_client_grad_list=raw_client_model_or_grad_list,
                    extra_auxiliary_info=self.get_model_params(),
                )
            if FederatedLearningAttacker.get_instance().is_model_attack():
                raw_client_model_or_grad_list = FederatedLearningAttacker.get_instance().attack_model(
                    raw_client_grad_list=raw_client_model_or_grad_list,
                    extra_auxiliary_info=self.get_model_params(),
                )
            client_idxs = [i for i in range(len(raw_client_model_or_grad_list))]
            if FederatedLearningDefender.get_instance().is_defense_enabled():
                raw_client_model_or_grad_list = FederatedLearningDefender.get_instance().defend_before_aggregation(
                    raw_client_grad_list=raw_client_model_or_grad_list,
                    extra_auxiliary_info=self.get_model_params(),
                )
                client_idxs = FederatedLearningDefender.get_instance().get_benign_client_idxs(client_idxs=client_idxs)

            return raw_client_model_or_grad_list, client_idxs

    def aggregate(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        if FederatedLearningFHE.get_instance().is_fhe_enabled():
            logging.info(" ---- aggregating models using homomorphic encryption ----")
            return FederatedLearningFHE.get_instance().fhe_fedavg(raw_client_model_or_grad_list)
        else:
            if FederatedLearningDefender.get_instance().is_defense_enabled():
                return FederatedLearningDefender.get_instance().defend_on_aggregation(
                    raw_client_grad_list=raw_client_model_or_grad_list,
                    base_aggregation_func=FederatedLearningAggOperator.agg,
                    extra_auxiliary_info=self.get_model_params(),
                )
            if FederatedLearningDifferentialPrivacy.get_instance().to_compute_params_in_aggregation_enabled():
                FederatedLearningDifferentialPrivacy.get_instance().set_params_for_dp(raw_client_model_or_grad_list)
            return FederatedLearningAggOperator.agg(self.args, raw_client_model_or_grad_list)

    def on_after_aggregation(self, aggregated_model_or_grad: OrderedDict) -> OrderedDict:
        if FederatedLearningFHE.get_instance().is_fhe_enabled():
            logging.info(" ---- finish aggregating encrypted global model ----")
            dec_aggregated_model_or_grad = aggregated_model_or_grad
            return dec_aggregated_model_or_grad
        else:
            if FederatedLearningDifferentialPrivacy.get_instance().is_global_dp_enabled():
                logging.info("-----add central DP noise ----")
                aggregated_model_or_grad = FederatedLearningDifferentialPrivacy.get_instance().add_global_noise(
                    aggregated_model_or_grad
                )
            if FederatedLearningDefender.get_instance().is_defense_enabled():
                aggregated_model_or_grad = FederatedLearningDefender.get_instance().defend_after_aggregation(aggregated_model_or_grad)
            return aggregated_model_or_grad

    def assess_contribution(self):
        if self.contribution_assessor_mgr is None:
            return
        # TODO: start to run contribution assessment in an independent python process
        client_num_per_round = len(Context().get(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND))
        client_index_for_this_round = Context().get(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND)
        local_weights_from_clients = Context().get(Context.KEY_CLIENT_MODEL_LIST)

        metric_results_in_the_last_round = Context().get(Context.KEY_METRICS_ON_LAST_ROUND)
        (acc_on_last_round, _, _, _) = metric_results_in_the_last_round
        metric_results_on_aggregated_model = Context().get(Context.KEY_METRICS_ON_AGGREGATED_MODEL)
        (acc_on_aggregated_model, _, _, _) = metric_results_on_aggregated_model
        test_data = Context().get(Context.KEY_TEST_DATA)
        validation_func = self.test
        self.contribution_assessor_mgr.run(
            client_num_per_round,
            client_index_for_this_round,
            FederatedLearningAggOperator.agg,
            local_weights_from_clients,
            acc_on_last_round,
            acc_on_aggregated_model,
            test_data,
            validation_func,
            self.args.device,
        )

        if self.args.round_idx == self.args.comm_round - 1:
            self.final_contribution_assigment_dict = self.contribution_assessor_mgr.get_final_contribution_assignment()
            logging.info(
                "self.final_contribution_assigment_dict = {}".format(self.final_contribution_assigment_dict))

    @abstractmethod
    def test(self, test_data, device, args):
        pass

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        pass
