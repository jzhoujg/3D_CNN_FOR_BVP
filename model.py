import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

class Three_D_Model(nn.Module):

    def __init__(self, batch_size = 128, num_channels = 1,num_frames = 34, height = 20, width = 20,device="cpu", Trans=False):

        super(Three_D_Model, self).__init__()

        self.features_1 = nn.Sequential(nn.Conv3d(in_channels=1,out_channels=64,kernel_size= (3,3,3),stride=1, padding=1),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size = (2,2,2), stride=2),
        )

        self.features_2 = nn.Sequential(nn.Conv3d(in_channels=64,out_channels=64,kernel_size= (3,3,3),stride=1, padding=1),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size = (2,2,2), stride=2),
        )

        self.features_3 = nn.Sequential(
                                    nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
        )


        self.head = nn.Sequential(
                                    nn.Linear(1024, 128),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(64,  6)
        )

        self.apply(_init_vit_weights)
        self.batchsize = batch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.device = device
        self.trans = Trans


    def ReadBVPmap(self,BVP_map):
        row, col = 6, 6
        BVP_frames = BVP_map[:,:, 0:20, 0:20]
        in_batch = BVP_map.size()[0]
        BVP_TEMP = torch.zeros(in_batch,1,self.num_frames,self.height,self.width,device=self.device)
        for i in range(row):
            for j in range(col):
                one_pic = BVP_map[:, :, 0 + 20 * j:20 + 20 * j, 0 + 20 * i:20 + 20 * i]
                if self.trans:
                    one_pic = transforms.RandomHorizontalFlip()(one_pic)
                    one_pic = transforms.Resize(21)(one_pic)
                    one_pic = transforms.RandomCrop((20, 20))(one_pic)
                #     #                                transforms.CenterCrop(224),

                BVP_frames = torch.cat((BVP_frames, one_pic), dim=1)

        BVP_TEMP[:, 0, :, :, :] = BVP_frames[:, 2:36, :, :]



        return BVP_TEMP


    def forward(self,x):
        x = self.ReadBVPmap(x)
        x = self.features_1(x)
        x = self.features_2(x)
        x = self.features_3(x)
        x = torch.flatten(x,1)
        x = self.head(x)

        return x

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

if __name__ == '__main__':
    image = Image.open('1.png')
    image = transforms.ToTensor()(image)
    temp = torch.zeros((1,1,120,120))
    temp[0,0,:,:] = image[:,:]
    model = Three_D_Model(batch_size=1)
    print(model.forward(temp))

