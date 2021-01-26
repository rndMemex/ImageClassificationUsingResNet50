

#==============================================================================================
# This is code was started at the Huawei AI Challenge at  The University of Cambridge Hackathon (HexCambridge 2021) 
# ============================================================================
import os
import argparse
from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.resnet import resnet50 as resnet
from mindspore.nn.metrics import Accuracy
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C



def create_dataset(data_home, do_train, batch_size=32, repeat_num=1):
    """
    create a train or evaluate cifar10 dataset for resnet50
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend

    Returns:
        dataset
    """
    # define dataset
    cifar_ds = ds.Cifar10Dataset(data_home)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4)) # padding_mode default CONSTANT
    random_horizontal_op = C.RandomHorizontalFlip()
    resize_op = C.Resize((resize_height, resize_width)) # interpolation default BILINEAR
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = C.HWC2CHW()
    type_cast_op = C2.TypeCast(mstype.int32)

    c_trans = []
    if do_train:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label")
    cifar_ds = cifar_ds.map(operations=c_trans, input_columns="image")


    # apply DatasetOps
    # buffer_size = 10000
    # apply shuffle operations
    cifar_ds = cifar_ds.shuffle(buffer_size=10)

    # apply batch operations
    #cifar_ds = cifar_ds.batch(batch_size=args.batch_size, drop_remainder=True) #fix this
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True) #fix this

    # apply repeat operations
    cifar_ds = cifar_ds.repeat(repeat_num)

    return cifar_ds


def test_net(network,model,data_home, ckpoint):
    """define the evaluation method"""
    print("============== Starting Testing ==============")
    #load the saved model for evaluation
    param_dict = load_checkpoint(ckpoint) #MindSpore_Resnet50_cifar10.ckpt
    print("STEP 1: LOAD CHECKPOINT")
    #load parameter to the network
    load_param_into_net(network, param_dict)
    print("STEP 2: LOAD PARAM INTO NETWORK: ResNet50")
    #load testing dataset
    ds_eval = create_dataset(os.path.join(data_home, "cifar-10-verify-bin"), do_train=False, batch_size=32,repeat_num=1) # cifar-10-verify-bin
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MindSpore CIFAR-10 Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--checkpoint_path', type=str, default="MindSpore_Resnet50_cifar10.ckpt", help='Pretrained checkpoint path')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    dataset_sink_mode = not args.device_target == "CPU"

    # define net
    net = resnet(class_num=10) #if you wish to consider other module you can pass also this argument
    # ckpoint = args.checkpoint_path
    # define loss, model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    #define cifar-10 path
    cifar_path = "./CIFAR-10"
    # define model
    model = Model(net, loss_fn=loss, metrics={"Accuracy": Accuracy()})
    # eval model
    test_net(net, model, cifar_path, args.checkpoint_path)


    #     # config for resent50, cifar10
    # config1 = ed({
    #     "class_num": 10,
    #     "batch_size": 32,
    #     "loss_scale": 1024,
    #     "momentum": 0.9,
    #     "weight_decay": 1e-4,
    #     "epoch_size": 90,
    #     "pretrain_epoch_size": 0,
    #     "save_checkpoint": True,
    #     "save_checkpoint_epochs": 5,
    #     "keep_checkpoint_max": 10,
    #     "save_checkpoint_path": "./",
    #     "warmup_epochs": 5,
    #     "lr_decay_mode": "poly",
    #     "lr_init": 0.01,
    #     "lr_end": 0.00001,
    #     "lr_max": 0.1
    # })
