from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from optimizer import PruneAdam
from model import LeNet, AlexNet, ResNet, ResidualBlock, VGG ,L_softmax,MLP
from utils import regularized_nll_loss, admm_loss, \
    initialize_Z_and_U, update_X, update_Z, update_Z_l1, update_U, \
    print_convergence, print_prune, apply_prune, apply_l1_prune,update_Z_SCAD,update_Z_l0,apply_l0_prune,updata_Z_Prox_glarho,\
    apply_rscad_prune
from torchvision import datasets, transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
import  numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle



#------------------画出权重的分布图-------------------------------------
def plot_weights(model):
    modules = [module for module in model.modules()]
    num_sub_plot = 0
    for i, layer in enumerate(modules):
        if hasattr(layer, 'weight'):
            plt.subplot(181+num_sub_plot)
            w = layer.weight.data
            w_one_dim = w.cpu().numpy().flatten()
            plt.hist(w_one_dim[w_one_dim!=0], bins=50)
            num_sub_plot += 1
    plt.show()

def train(args, model, device, train_loader, test_loader, optimizer):
    loss_iter = []
    for epoch in range(args.num_pre_epochs):

        print('Pre epoch: {}'.format(epoch + 1))
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = regularized_nll_loss(args, model, output, target)
            loss_iter.append(loss)
            loss.backward()
            optimizer.step()
        test(args, model, device, test_loader)

    Z, U = initialize_Z_and_U(model)   #初始化 Z，U
    A = np.zeros((args.idx,args.num_epochs))
    for epoch in range(args.num_epochs):
        model.train()
        print('Epoch: {}'.format(epoch + 1))
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = admm_loss(args, device, model, Z, U, output, target)
            loss.backward()
            optimizer.step()
        X = update_X(model)  #更新X
        #Z的更新根据正则项来选择
        if (args.l1):
            Z = update_Z_l1(X,U,args)
        elif (args.l0):
            Z= update_Z_l0(X,U,args)
        elif (args.SCAD):
            Z = update_Z_SCAD(X,U,args)
        elif (args.rscad):
            print('use rscad updata z')
            Z = updata_Z_Prox_glarho(X,U,args)
        else:
            Z = update_Z(X,U,args)
          #根据稀疏项 选择跟新Z 方式
        U = update_U(U, X, Z)

        if not args.test_lamda:
            a = print_convergence(model, X, Z)
            for i in range(args.idx):
                A[i,epoch] = a[i]

        test(args, model, device, test_loader)
    return  A

def test(args, model, device, test_loader):                              #测试函数
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def retrain(args, model, mask, device, train_loader, test_loader, optimizer):
    for epoch in range(args.num_re_epochs):
        print('Re epoch: {}'.format(epoch + 1))
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.prune_step(mask)

        test(args, model, device, test_loader)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default="mnist_logistic", choices=["mnist","mnist_logistic", "mnist_MLP", \
                                                                                   "cifar10_AlexNet",\
                                                                         "cifar10_ResNet", "cifar10_VGG"],
                        metavar='D', help='training dataset (mnist or cifar10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--percent', type=list, default=[0.8, 0.92, 0.991, 0.93],
                        metavar='P', help='pruning percentage (default: 0.8)')
    parser.add_argument('--alpha', type=float, default=1e-4, metavar='L',
                        help='l2 norm weight (default: none')
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

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == "mnist":
        args.num_pre_epochs = 3
        args.num_epochs = 30
        args.num_re_epochs = 1
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    elif args.dataset == "mnist_logistic":
        args.num_pre_epochs = 5
        args.num_epochs = 80
        args.num_re_epochs = 1
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    elif args.dataset == "mnist_MLP":
        args.num_pre_epochs = 2
        args.num_epochs = 50
        args.num_re_epochs = 1
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    else:
        args.percent = [0.8, 0.92, 0.93, 0.94, 0.95, 0.99, 0.99, 0.93]
        args.num_pre_epochs = 20
        args.num_epochs = 50
        args.num_re_epochs = 2
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4), #数据增强 -将图片转化为周围加上4圈0 再裁剪为32x32
                                 transforms.RandomHorizontalFlip(),   #图像翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=args.batch_size, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=args.test_batch_size, **kwargs)
    args.test_lamda = False
    args.plot_convergence = True
    if args.test_lamda:
        if args.l1:
            lamda = [2.5e-4,3e-4,3.5e-4,4e-4,4.5e-4,5e-4, 5.5e-4, 6e-4, 6.5e-4, 7e-4, 7.5e-4]
        elif args.l0:
            lamda = [2.5e-4, 3e-4, 3.5e-4, 4e-4, 4.5e-4, 5e-4, 5.5e-4, 6e-4, 6.5e-4, 7e-4, 7.5e-4]
        else:
            lamda = [4e-4,4.5e-4,5e-4, 5.5e-4, 6e-4, 6.5e-4, 7e-4, 7.5e-4, 8e-4, 8.5e-4, 9e-4, 9.5e-4, 1e-3]

        for i in range(0,len(lamda),1):
            args.alpha =lamda[i]
            print("**********************the test lamda*****************************")
            print('\nThe test lamda: {:.6f}\n'.format(args.alpha))
            #模型选择
            if args.dataset == "mnist":
                model = LeNet().to(device)
            elif args.dataset =="mnist_logistic" :
                model = L_softmax().to(device)
            elif args.dataset =="mnist_MLP" :
                model = MLP().to(device)
            elif args.dataset == "cifar10_AlexNet":
                model = AlexNet().to(device)
            elif args.dataset == "cifar10_ResNet":
                model = ResNet(ResidualBlock).to(device)
            else:
                model = VGG().to(device)


            optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)

            train(args, model, device, train_loader, test_loader, optimizer)

            if (args.l1):
                mask = apply_l1_prune(model, device, args)
            elif (args.l0):
                mask = apply_l0_prune(model, device, args)
            elif (args.SCAD):
                mask = apply_l1_prune(model, device, args)
            elif (args.rscad):
                mask = apply_rscad_prune(model, device, args)
            else:
                mask = apply_prune(model, device, args)

            print_prune(model)

            test(args, model, device, test_loader)
            retrain(args, model, mask, device, train_loader, test_loader, optimizer)
    else:

        if args.dataset == "mnist":
            print('\n*********** The test model is LeNet and lamda = %f and dataset is minst************' % args.alpha)
            model = LeNet().to(device)
            args.idx = 4
        elif args.dataset == "mnist_logistic":
            print('\n*********** The test model is logistic and lamda = %f and dataset is minst************' % args.alpha)
            args.idx = 1
            model = L_softmax().to(device)
        elif args.dataset == "mnist_MLP":
            print('\n*********** The test model is MLP and lamda = %f and dataset is minst************' % args.alpha)
            args.idx = 3
            model = MLP().to(device)
        elif args.dataset == "cifar10_AlexNet":
            print('\n*********** The test model is AlexNet and lamda = %f and dataset is cifar10************' % args.alpha)
            model = AlexNet().to(device)
            args.idx = 8
        elif args.dataset == "cifar10_ResNet":
            print('\n*********** The test model is ResNet and lamda = %f and dataset is cifar10************' % args.alpha)
            model = ResNet(ResidualBlock).to(device)
        else:
            print('\n*********** The test model is VGG-16 and lamda = %f and dataset is cifar10************' % args.alpha)
            model = VGG().to(device)
            args.idx = 9
        optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
        A = train(args, model, device, train_loader, test_loader, optimizer)

        if args.plot_convergence:
            color = ['purple','red','blue','yellow','cyan', 'green','magenta','black','gray','hotpink']
            marke =['p','*','o','x','s','v','h','|','d','+']
            plt.figure
            aixs_x = np.arange(1,args.num_epochs+1, 1)
            for i in range(args.idx):
                plt.plot(aixs_x,sorted(A[i], reverse= True),color = color[i],linewidth = 1, \
                         label = 'conv1',linestyle = '--',marker = marke[i])
            plt.xlabel('iterations')
            plt.ylabel('error')
            plt.title('||theta^k+1-z^k+1||')
            if args.dataset == "mnist":
                plt.legend(['conv1', 'conv2', 'fc1', 'fc2'])
            elif args.dataset == "mnist_MLP":
                plt.legend(['fc1', 'fc2', 'fc3'])
            elif args.dataset == "mnist_logistic":
                plt.legend(['weight'])
            elif args.dataset == "cifar10_AlexNet":
                plt.legend(['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3'])
            else:
                plt.legend(['conv1','conv2','conv3','conv4','conv5','conv6','conv7','conv8','fc1'])

            plt.savefig("convergence_logistic10.jpg",dpi=600)

        if (args.l1):
            mask = apply_l1_prune(model, device, args)
        elif (args.l0):
            mask = apply_l0_prune(model, device, args)
        elif (args.SCAD):
            mask = apply_l1_prune(model, device, args)
        elif (args.rscad):
            mask = apply_rscad_prune(model, device, args)
        else:
            mask = apply_prune(model, device, args)

        #pickle.dump(model, open("pruning_model.dat", "wb"))
        torch.save(model, 'pruning_modelnet_logistic10.pkl')
        print_prune(model)
        test(args, model, device, test_loader)
        retrain(args, model, mask, device, train_loader, test_loader, optimizer)

if __name__ == "__main__":
    main()