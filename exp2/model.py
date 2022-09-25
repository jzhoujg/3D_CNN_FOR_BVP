import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import cv2

class Spatial_model(nn.Module):

    def __init__(self, batch_size=128, num_channels=1,num_frames=34, height=20, width=20,device="cpu", Trans=False):

        super(Spatial_model, self).__init__()

        self.features_1 = nn.Sequential(nn.Conv2d(in_channels=20,out_channels=16,kernel_size= (2,2),stride=1, padding=1),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size = (2,2), stride=2),
        )



        self.head = nn.Sequential(
                                    nn.Linear(1600,128),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Dropout(0.6),
                                    nn.Linear(64,  6)
        )

        # self.apply(_init_vit_weights)
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
        BVP_TEMP = torch.zeros(in_batch,self.num_frames,self.height,self.width,device=self.device)
        for i in range(row):
            for j in range(col):
                one_pic = BVP_map[:, :, 0 + 20 * i:20 + 20 * i, 0 + 20 * j:20 + 20 * j]

                if self.trans:
                    one_pic = transforms.RandomHorizontalFlip()(one_pic)
                    one_pic = transforms.Resize(21)(one_pic)
                    one_pic = transforms.RandomCrop((20, 20))(one_pic)
                #     #                                transforms.CenterCrop(224),


                BVP_frames = torch.cat((BVP_frames, one_pic), dim=1)
                # print(one_pic)


        BVP_TEMP[:, :, :, :] = BVP_frames[:, 2:36, :, :]
        BVP_TEMP = torch.transpose(BVP_TEMP,1,2)

        return BVP_TEMP


    def forward(self,x):
        x = self.ReadBVPmap(x)
        x = self.features_1(x)
        # x = self.features_2(x)
        # x = self.features_2(x)
        # x = self.features_3(x)
        # x = torch.flatten(x,1)
        # x = self.head(x)

        return x


class Temporal_model(nn.Module):

    def __init__(self, batch_size = 128, num_channels = 1,num_frames = 34, height = 20, width = 20,device="cpu", Trans=False):
        super(Temporal_model, self).__init__()


        self.features_1 = nn.Sequential(nn.Conv2d(in_channels=20,out_channels=16,kernel_size= (2,2),stride=1, padding=1),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size = (2,2), stride=2),
        )


        self.head = nn.Sequential(
                                    nn.Linear(1600,128),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Dropout(0.6),
                                    nn.Linear(64,  6)
        )

        # self.apply(_init_vit_weights)
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
        BVP_TEMP = torch.zeros(in_batch,self.num_frames,self.height,self.width,device=self.device)
        BVP_RES = torch.zeros(in_batch, self.num_frames-1, self.height, self.width, device=self.device)
        for i in range(row):
            for j in range(col):
                one_pic = BVP_map[:, :, 0 + 20 * i:20 + 20 * i, 0 + 20 * j:20 + 20 * j]

                if self.trans:
                    one_pic = transforms.RandomHorizontalFlip()(one_pic)
                    one_pic = transforms.Resize(21)(one_pic)
                    one_pic = transforms.RandomCrop((20, 20))(one_pic)
                #     #                                transforms.CenterCrop(224),


                BVP_frames = torch.cat((BVP_frames, one_pic), dim=1)
                # print(one_pic)


        BVP_TEMP[:, :, :, :] = BVP_frames[:, 2:36, :, :]
        BVP_TEMP = torch.transpose(BVP_TEMP,1,3)
        # for m in range(self.num_frames -1):
        #     BVP_RES[:,m,:,:] =abs(BVP_TEMP[:,m+1, :, :]-BVP_TEMP[:,m, :, :])
        #     #BVP_RES[:, m, :, :] = transforms.Normalize([0.5],[0.5])(BVP_RES[:,m,:,:])




        return BVP_TEMP

    # def ReadBVPmap_O(self,BVP_map):
    #     row, col = 1, 34
    #     BVP_frames = BVP_map[:,:, 0:20, 0:20]
    #     in_batch = BVP_map.size()[0]
    #     BVP_TEMP = torch.zeros(in_batch,self.num_frames+1,self.height,self.width,device=self.device)
    #     for i in range(row):
    #         for j in range(col):
    #             one_pic = BVP_map[:, :, 0 + 20 * i:20 + 20 * i, 0 + 20 * j:20 + 20 * j]
    #
    #             if self.trans:
    #                 one_pic = transforms.RandomHorizontalFlip()(one_pic)
    #                 one_pic = transforms.Resize(21)(one_pic)
    #                 one_pic = transforms.RandomCrop((20, 20))(one_pic)
    #             #     #                                transforms.CenterCrop(224),
    #
    #
    #             BVP_frames = torch.cat((BVP_frames, one_pic), dim=1)
    #
    #
    #     BVP_TEMP[:, :, :, :] = BVP_frames[:, 1:, :, :]
    #
    #
    #
    #
    #     return BVP_TEMP


    def forward(self,x):
        x = self.ReadBVPmap(x)
        x = self.features_1(x)
        x = torch.transpose(x,2,3)
        # x = self.features_2(x)
        # x = torch.flatten(x,1)
        # x = self.head(x)

        return x


class Two_Stream_model(nn.Module):

    def __init__(self, batch_size = 128, num_channels = 1,num_frames = 34, height = 20, width = 20,device="cpu", Trans=False):
        super(Two_Stream_model, self).__init__()

        self.funsion_1 = nn.Sequential(nn.Conv3d(in_channels=2,out_channels=64,kernel_size= (3,3,3),stride=1, padding=1),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size = (2,2,2), stride=2),

                                     nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=1,
                                               padding=1),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
        )

        self.head_1 = nn.Sequential(
                                    nn.Linear(2048, 128),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Dropout(0.6),
                                    nn.Linear(64,  6)
        )



        self.funsion_2 = nn.Sequential(nn.Conv3d(in_channels=2,out_channels=64,kernel_size= (3,3,3),stride=1, padding=1),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size = (2,2,2), stride=2),

                                     nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=1,
                                               padding=1),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(),
                                     nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
        )

        self.head_2 = nn.Sequential(
                                    nn.Linear(2048, 128),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Dropout(0.6),
                                    nn.Linear(64,  5)
        )

        # self.apply(_init_vit_weights)
        self.batchsize = batch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.device = device
        self.trans = Trans
        self.temp = Temporal_model(device=device)
        self.spat = Spatial_model(device=device)



    def forward(self,x):

        x_t = self.temp.forward(x)
        x_s = self.spat.forward(x)

        x_t = torch.transpose(x_t,1,2)
        x_s = torch.transpose(x_s,1,2)
        in_batch = x.size()[0]
        temp = torch.zeros((in_batch,2,17,16,10),device=self.device)

        temp[:,0,:,:,:] = x_s[:,:,:,:]
        temp[:,1,:,:,:] = x_t[:,:,:,:]


        x = self.funsion_1(temp)
        y = self.funsion_2(temp)

        x = torch.flatten(x,1)
        y = torch.flatten(y,1)
        x = self.head_1(x)
        y = self.head_2(y)
        res = (x,y)

        return res

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
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
    model = Two_Stream_model()
    print(model.forward(temp)[1])