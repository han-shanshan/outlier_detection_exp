import logging
import federated_learning
from federated_learning import FederatedLearningRunner
from federated_learning.model.cv.resnet import resnet20
import wandb



if __name__ == "__main__":
    args = federated_learning.init()

    if args.enable_wandb:
        args.wandb_obj = wandb.init(
            entity="federated_learning", project="federated_learningSecurity", name="lr_security", config=args
        )

    # init device
    device = federated_learning.device.get_device(args)

    # load data
    dataset, output_dim = federated_learning.data.load(args)

    # load model
    model = federated_learning.model.create(args, output_dim)

    # start training
    federated_learning_runner = FederatedLearningRunner(args, device, dataset, model)
    federated_learning_runner.run()
