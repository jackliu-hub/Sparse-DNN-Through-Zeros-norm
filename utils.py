import torch
import torch.nn.functional as F
import numpy as np

#二范数正则项
def regularized_nll_loss(args, model, output, target):  #此函数为参数加上二范数正则项
    index = 0
    loss = F.nll_loss(output, target)
    if args.l2:
        for name, param in model.named_parameters():
            if name.split('.')[-1] == "weight":
                loss += args.alpha * param.norm()
                index += 1
    return loss

#此函数为 admm训练第一步
def admm_loss(args, device, model, Z, U, output, target):
    idx = 0
    loss = F.nll_loss(output, target)
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            u = U[idx].to(device)
            z = Z[idx].to(device)
            loss += args.rho / 2 * (param - z + u).norm()
            if args.l2:
                loss += args.alpha * param.norm()          #如果要加上二范数正则
            idx += 1
    return loss

 #初始化Z和U
def initialize_Z_and_U(model):
    Z = ()
    U = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            Z += (param.detach().cpu().clone(),)
            U += (torch.zeros_like(param).cpu(),)
    return Z, U

#更新x
def update_X(model):
    X = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            X += (param.detach().cpu().clone(),)
    return X

#投影正则更新z
def update_Z(X, U, args):
    new_Z = ()
    idx = 0
    for x, u in zip(X, U):
        z = x + u   #执行合并那一步
        pcen = np.percentile(abs(z), 100*args.percent[idx])   #投影率计算
        under_threshold = abs(z) < pcen
        z.data[under_threshold] = 0
        new_Z += (z,)
        idx += 1
    return new_Z

#l1 正则跟新 z
def update_Z_l1(X, U, args):   #一范数正则
    new_Z = ()
    delta = args.alpha / args.rho
    for x, u in zip(X, U):
        z = x + u
        new_z = z.clone()
        if (z > delta).sum() != 0:
            new_z[z > delta] = z[z > delta] - delta
        if (z < -delta).sum() != 0:
            new_z[z < -delta] = z[z < -delta] + delta
        if (abs(z) <= delta).sum() != 0:
            new_z[abs(z) <= delta] = 0
        new_Z += (new_z,)
    return new_Z

#零范式正则 更新z
def update_Z_l0(X, U ,args):
    new_Z = ()
    delta = np.sqrt(2*args.alpha / args.rho)
    for x, u in zip(X, U):
        z = x + u
        new_z = z.clone()
        if (abs(z) > delta).sum() != 0:
            new_z[abs(z) > delta] = z[abs(z) > delta]
        if (abs(z) <= delta).sum() != 0:
            new_z[abs(z) <= delta] = 0
        new_Z += (new_z,)
    return new_Z

def soft_throd(X, lr):
    y = X.clone()
    index1 = X > lr
    y[index1] -= lr
    index2 = X < (-lr)
    y[index2] += lr
    index3 = abs(X) <= lr
    y[index3] = 0
    return y

#scad正则更新Z
def update_Z_SCAD(X, U, args):
    new_Z = ()
    delta = args.alpha / args.rho
    a = args.a
    for x, u in zip(X, U):
        z = x + u
        new_z = z.clone()
        if (abs(z) <= 2*delta).sum() != 0:
            new_z[abs(z)< 2*delta ] = soft_throd(z[abs(z)< 2*delta], delta)
        if ((2*delta < abs(z)) & (abs(z) <= a*delta)).sum() != 0:
            index = (2*delta < abs(z)) & (abs(z) <= a*delta)
            new_z[index] = ((a-1)*z[index]-np.sign(z[index])*a*delta)/(a-2)
        if (abs(z) > a*delta).sum() != 0:
            new_z[abs(z) > a*delta] = new_z[abs(z) > a*delta]
        new_Z += (new_z,)

    return new_Z


#修正scad正则
def updata_Z_Prox_glarho(X,U,args):
    new_Z = ()
    delta = args.alpha / args.rho
    a = args.a
    rho =1.5
    sigma = 1/delta
    asigma = 2*(a - 1)*sigma
    arho = ((a + 1)*rho)
    fd1 = delta
    fd2 = fd1 + 2/arho
    fd3 = 2*a/arho
    for x, u in zip(X, U):
        z = x + u
        new_z = z.clone()
        if ((abs(z)>fd1) & (abs(z)<=fd2)).sum() != 0:
            index = (abs(z)>fd1) & (abs(z)<=fd2)
            new_z[index] =  z[index]-delta*np.sign(z[index])
        if ((abs(z)>fd2)&(abs(z)<=fd3)).sum() != 0:
            index = (abs(z)>fd2)&(abs(z)<=fd3)
            new_z[index] = (asigma*z[index]-2*a*np.sign(z[index]))/(asigma-arho)
        if (abs(z) > fd3).sum() != 0:
            new_z[abs(z) > fd3] = z[abs(z) > fd3]
        if (abs(z) <= fd1).sum() != 0:
            new_z[abs(z)<fd1] = 0
        new_Z += (new_z,)
    return new_Z

#更新 乘子U
def update_U(U, X, Z):
    new_U = ()
    for u, x, z in zip(U, X, Z):
        new_u = u + (x - z)
        new_U += (new_u,)
    return new_U

#投影正则生成掩码
def prune_weight(weight, device, percent):   #生成掩码矩阵
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()
    pcen = np.percentile(abs(weight_numpy), 100*percent)
    under_threshold = abs(weight_numpy) < pcen
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= pcen).to(device)
    return mask

#l1正则生成掩码
def prune_l1_weight(weight, device, delta):
    weight_numpy = weight.detach().cpu().numpy()
    under_threshold = abs(weight_numpy) < delta
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= delta).to(device)
    return mask

#l0正则生成掩码
def prune_l0_weight(weight, device, delta):
    weight_numpy = weight.detach().cpu().numpy()
    under_threshold = abs(weight_numpy) < delta
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= delta).to(device)
    return mask

def prune_rscad_weight(weight, device, delta):
    weight_numpy = weight.detach().cpu().numpy()
    under_threshold = abs(weight_numpy) < delta
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= delta).to(device)
    return mask

#投影正则剪枝函数
def apply_prune(model, device, args):
    # returns dictionary of non_zero_values' indices
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_weight(param, device, args.percent[idx])
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
            idx += 1
    return dict_mask

#l1正则剪枝函数
def apply_l1_prune(model, device, args):
    delta = args.alpha / args.rho
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_l1_weight(param, device, delta)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask

#l0正则剪枝函数
def apply_l0_prune(model, device, args):
    delta = np.sqrt(2*args.alpha / args.rho)
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_l0_weight(param, device, delta)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask

def apply_rscad_prune(model, device, args):
    delta = np.sqrt(2*args.alpha / args.rho)
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_rscad_weight(param, device, delta)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask

#输出收敛速率
def print_convergence(model, X, Z):
    idx = 0
    a = []
    print("normalized norm of (weight - projection)")
    for name, _ in model.named_parameters():
        if name.split('.')[-1] == "weight":
            x, z = X[idx], Z[idx]
            print("({}): {:.4f}".format(name, (x-z).norm().item()/ x.norm().item()))
            a.append((x-z).norm().item()/ x.norm().item())
            idx += 1

    return a
#剪枝率
def print_prune(model):
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))
