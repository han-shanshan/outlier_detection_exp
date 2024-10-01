#!/usr/bin/env bash

WORKER_NUM=$1

TASK_NAMES=(
    "40_attack_rounds_femnist_cnn_byzantine_random_1adv_trimmed_mean"
    # "label_flipping_0_1_cnn_femnist_3sigma"
#     "40_attack_rounds_femnist_cnn_byzantine_random_1adv"
#     # "10_attack_rounds_femnist_cnn_byzantine_random_1adv_3sigma"
#     # "40_attack_rounds_femnist_cnn_byzantine_random_1adv_3sigma"
#     # "70_attack_rounds_femnist_cnn_byzantine_random_1adv_3sigma"
#     "40_attack_rounds_femnist_cnn_byzantine_random_1adv_foolsgold"
#  "40_attack_rounds_femnist_cnn_byzantine_random_1adv_krum_m5"
#  "40_attack_rounds_femnist_cnn_byzantine_random_1adv_rfa"
#  "40_attack_rounds_femnist_cnn_byzantine_random_1adv_rlr"
    # "label_flipping_0_1_cnn_femnist_0_1"
# "label_flipping_0_1_cnn_femnist_3sigma"
# "label_flipping_0_1_cnn_femnist_krum_m5"
# "label_flipping_0_1_cnn_femnist_rlr"
# "label_flipping_0_1_femnist_cnn_1adv_bucketing"
# "label_flipping_0_1_femnist_cnn_1adv_trimmed_mean"
# "label_flipping_0_1_femnist_cnn_foolsgold"
# "label_flipping_0_1_femnist_cnn_rfa"
# "label_flipping_0_1_cnn_femnist_0_1"
# "label_flipping_0_1_cnn_femnist_3sigma"
# "label_flipping_0_1_cnn_femnist_krum_m5"
# "label_flipping_0_1_cnn_femnist_rlr"
# "femnist_cnn_model_replacement_1adv_rlr_threshold_0_5"
# "femnist_cnn_model_replacement_1adv_rlr_threshold_10"
# "femnist_cnn_model_replacement_1adv_rlr_threshold_1"
# "femnist_cnn_model_replacement_1adv_rlr_threshold_4"

# "femnist_cnn_byzantine_random_1adv_rlr"
# "femnist_cnn_byzantine_random_1adv_rfa"
# "femnist_cnn_byzantine_random_1adv"
# "femnist_cnn_byzantine_random_1adv_krum_m5"
# "femnist_cnn_benign"
)  

IDXs=("1")

DATA_PARTITION_TYPE="hetero"

CLIENT_NUM="10"

for IDX in "${IDXs[@]}"
do
    for TASK_NAME in "${TASK_NAMES[@]}"
    do
        LOG_FILE="logs/CNN/${CLIENT_NUM}clients_${DATA_PARTITION_TYPE}/${DATA_PARTITION_TYPE}_CNN_${TASK_NAME}_${CLIENT_NUM}clients_${IDX}.log"

        PROCESS_NUM=`expr $WORKER_NUM + 1`
        echo $PROCESS_NUM $TASK_NAME $IDX

        hostname > mpi_host_file

        CONFIG="config/CNN/${CLIENT_NUM}clients_${DATA_PARTITION_TYPE}/${TASK_NAME}.yaml"

        mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
        python torch_mpi_cnn.py --cf $CONFIG > $LOG_FILE 2>&1
    done
done