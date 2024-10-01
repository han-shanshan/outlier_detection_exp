import logging
import federated_learning
from federated_learning import FedMLRunner
from federated_learning.model.cv.resnet import resnet56


def create_model():
    pre_trained_model_path = "config/CV/resnet56_10clients_hetero/resnet56_on_cifar10.pth"
    # model = resnet56(100, pretrained=False, path=pre_trained_model_path)
    model = resnet56(100, pretrained=False)
    logging.info("load model successfully")
    return model


if __name__ == "__main__":
    args = federated_learning.init()

    # init device
    device = federated_learning.device.get_device(args)

    # load data
    dataset, output_dim = federated_learning.data.load(args)

    # load model
    # model = federated_learning.model.create(args, output_dim)
    model = create_model()

    # start training
    federated_learning_runner = FedMLRunner(args, device, dataset, model)
    federated_learning_runner.run()
