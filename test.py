import torch
import torch.nn as nn

def prune(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack(binary, white_model, black_model, dataloader, epsilon_size, epochs, attack_name, rand=True):
    if attack_name == "cifar":
        mean, std = torch.Tensor([0.471, 0.448, 0.408]), torch.Tensor([0.234, 0.239, 0.242])
        epsilon = ((1 / 255) - mean) / std - ((0 / 255) - mean) / std
        epsilon = epsilon_size * epsilon.view(1,3,1,1).repeat(32, 1, 32, 32).cuda()
        clip_min, clip_max = ((0 / 255) - mean) / std, ((255 / 255) - mean) / std
        clip_min, clip_max = clip_min.view(1,3,1,1).repeat(32, 1, 32, 32).cuda(), clip_max.view(1,3,1,1).repeat(32, 1, 32, 32).cuda()

    elif attack_name == "mnist":
        mean, std = torch.Tensor([0.1307,]), torch.Tensor([0.3081,])
        epsilon = ((1 / 255) - mean) / std - ((0 / 255) - mean) / std
        epsilon = epsilon_size * epsilon.view(1,1,1,1).repeat(1, 1, 28, 28).cuda()
        clip_min, clip_max = ((0 / 255) - mean) / std, ((255 / 255) - mean) / std
        clip_min, clip_max = clip_min.view(1,1,1,1).repeat(32, 1, 28, 28).cuda(), clip_max.view(1,1,1,1).repeat(32, 1, 28, 28).cuda()

    loss_func = nn.CrossEntropyLoss()
    attack_num, num = 0., 0.
    for data,label in dataloader:
        data, label = data.cuda(), label.cuda()
        adv_data = data.detach().clone().cuda()
        if rand:
            adv_data = adv_data + torch.rand_like(adv_data, device = "cuda") / 500
        adv_data.requires_grad = True
        for epoch in range(epochs):
            loss_func(white_model(adv_data), label).backward()
            with torch.no_grad():
                if binary == False:
                    update_value = adv_data.grad / adv_data.grad.pow(2).sum(dim = [1,2,3]).sqrt().view(-1,1,1,1)
                elif binary == True:
                    update_value = 0.7*torch.sign(adv_data.grad)
                adv_data = adv_data + update_value
                adv_data = prune(adv_data, clip_min, clip_max)
                adv_data = prune(adv_data, data - epsilon, data + epsilon)
                adv_data.requires_grad = True

        attack_num += (black_model(adv_data).max(dim = 1)[1] != label).sum().item()
        num += label.size()[0]

    return attack_num / num * 100



def cifar():
    import torchvision
    import os

    mean, std = torch.Tensor([0.471, 0.448, 0.408]), torch.Tensor([0.234, 0.239, 0.242])
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean,std)])
    data = torchvision.datasets.CIFAR10(root="../data",train=False,download=False,transform=transform)
    loader = torch.utils.data.DataLoader(data,batch_size=32,shuffle=False,drop_last=True)
    import models.wideresnet as models
    white_model = models.WideResNet(num_classes=10).cuda()

    import models.mobilenet as BlackModel
    black_model = BlackModel.MobileNet().cuda()
    black_model.load_state_dict(torch.load("black_model/mobilenet.p")["net"]) # 
    black_model.eval()

    temp = torch.load(os.path.join('wideresnet_vs_mobilenet/result_1000', "model_best.pth.tar"))
    white_model.load_state_dict(temp['state_dict'])
    white_model.eval()

    trans = attack(False, white_model, black_model, loader, epsilon, attack_num, "cifar", True)




if __name__ == '__main__':
    cifar()


