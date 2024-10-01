import logging
import federated_learning
from federated_learning import FedMLRunner
from federated_learning.model.cv.resnet import resnet20


def create_model():
    """
    load pretrained model...
    please download the pre-trained weight file from
    https://github.com/FedML-AI/FedML/blob/federated_learning_v0.6_before_fundraising/federated_learning_api/model/cv/pretrained/CIFAR10/resnet56/best.pth
    and rename the file to ``resnet56_on_cifar10.pth''
    """
    model = resnet20(10, pretrained=False)
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
