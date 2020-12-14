import argparse



parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataset', type=str, default="mnist",
                    choices=["mnist", "mnist_logistic", "mnist_MLP", "cifar10_AlexNet", \
                             "cifar10_ResNet", "cifar10_VGG"],
                    metavar='D', help='training dataset (mnist or cifar10)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--percent', type=list, default=[0.8, 0.92, 0.991, 0.93],
                    metavar='P', help='pruning percentage (default: 0.8)')
parser.add_argument('--alpha', type=float, default=3e-4, metavar='L',
                    help='l2 norm weight (default: none)')
parser.add_argument('--a', type=float, default=3.7, metavar='F',
                    help='SCAD norm weight (default: 3.7)')
parser.add_argument('--rho', type=float, default=1e-2, metavar='R',
                    help='cardinality weight (default: 1e-2)')
parser.add_argument('--l1', default=False, action='store_true',
                    help='prune weights with l1 regularization instead of cardinality')
parser.add_argument('--l0', default=True, action='store_true',
                    help='prune weights with l0 regularization instead of cardinality')
parser.add_argument('--SCAD', default=False, action='store_true',
                    help='prune weights with SCAD regularization instead of cardinality')
parser.add_argument('--rscad', default=False, action='store_true',
                    help='prune weights with RSCAD regularization instead of cardinality')
parser.add_argument('--l2', default=False, action='store_true',
                    help='apply l2 regularization')
parser.add_argument('--num_pre_epochs', type=int, default=3, metavar='P',
                    help='number of epochs to pretrain (default: 3)')
parser.add_argument('--num_epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num_re_epochs', type=int, default=3, metavar='R',
                    help='number of epochs to retrain (default: 3)')
parser.add_argument('--num_test_epochs', type=int, default=10, metavar='m',
                    help='number of epochs to retrain (default: 3)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-2)')
parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='E',
                    help='adam epsilon (default: 1e-8)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()