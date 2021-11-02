import torch.nn as nn
import torchvision.models as models


class KernelNet(nn.Module):
    def __init__(self, blur_kernel_size, pretrain_resnet="resnet18"):
        super(KernelNet, self).__init__()
        self.blur_kernel_size = blur_kernel_size
        if pretrain_resnet == "resnet18":
            self.pretrain = models.resnet18(pretrained=True)
        elif pretrain_resnet == "resnet34":
            self.pretrain = models.resnet34(pretrained=True)
        for param in self.pretrain.parameters():
            param.requires_grad = False
        num_ftrs = self.pretrain.fc.out_features
        self.linear = nn.Linear(num_ftrs, self.blur_kernel_size ** 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        res = self.pretrain(x)
        res = self.linear(res)
        res = self.softmax(res)
        return res.view(-1, self.blur_kernel_size, self.blur_kernel_size)

# print("------Test KernelNet-------")
# kernel_net_model = KernelNet(25).to("cuda")
# result = kernel_net_model(torch.rand(4,3,64,64,device="cuda"))
# print(result.size())
