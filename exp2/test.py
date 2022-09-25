import PIL.ImageShow
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import cv2



def ReadBVPmap(BVP_map, Trans=True):


    row, col = 6, 6
    BVP_frames = BVP_map[:, :, 0:20, 0:20]
    BVP_TEMP = torch.zeros(1, 1, 34, 20, 20)

    for i in range(row):
        for j in range(col):
            one_pic = BVP_map[:, :, 0 + 20 * i:20 + 20 * i, 0 + 20 * j:20 + 20 * j]
            if Trans:
                one_pic = transforms.RandomHorizontalFlip()(one_pic)
                one_pic = transforms.Resize(21)(one_pic)
                one_pic = transforms.RandomCrop((20,20))(one_pic)
                #                                transforms.CenterCrop(224),


            BVP_frames = torch.cat((BVP_frames, one_pic), dim=1)
    BVP_TEMP[:, 0, :, :, :] = BVP_frames[:, 2:36, :, :]

    return BVP_TEMP



image = Image.open('1.png')
image = transforms.ToTensor()(image)

# image = transforms.Resize(122)(image)
# image = transforms.RandomCrop((120,120))(image)
# image = transforms.ToPILImage()(image)
# PIL.ImageShow.show(image)
temp = torch.zeros((1, 1, 120, 120))
temp[0, 0, :, :] = image[:, :]

res = ReadBVPmap(temp,True)



img_1 = torch.zeros((20,20))
img_2 = torch.zeros((20,20))
img_out = torch.zeros(20,20,3)

for m in range(33):

    img_1[:,:] = res[0,0,m,:,:]
    img_2[:,:] = res[0,0,m+1,:,:]
    if m==0:
        img_1 = img_1.numpy()
        img_2 = img_2.numpy()
    hsv = np.zeros((20,20,3),dtype=np.float32)
    hsv[1,...] = 255
    flow = cv2.calcOpticalFlowFarneback(img_1, img_2, None, 0.5, 1, 1, 5, 5, 1.2, 0)

    # prevImg nextImg version pyr_scale levels winsize iterations  poly_n poly_sigma flags
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    img_out = np.hstack([img_out,bgr])


cv2.imwrite('1_out3.png',img_out)

# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
#     "val": transforms.Compose([transforms.Resize(256),
#                                transforms.CenterCrop(224),
#                                transforms.ToTensor(),
#                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
#


# out1 = np.array(flow[:,:,0])
# out2 = np.array(flow[:,:,1])
# max_element = 0
# min_element = 0
#
#
# for i in range(20):
#     for j in range(20):
#         max_element = max(out1[i,j],max_element)
#         min_element = min(out1[i,j],min_element)
#
#
# out1 -= min_element
# out1 = out1*255/(max_element-min_element)

#
# print(out1)
# out1 = torch.from_numpy(out1)
# out2 = torch.from_numpy(out2)

# ou1 = transforms.Normalize([0.5],[0.5])(out1)