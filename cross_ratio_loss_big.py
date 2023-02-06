import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CrossRatioLossBig(nn.Module):
    def __init__(self, loss_type, include_geo, geo_loss_gamma_horz, geo_loss_gamma_vert):
        super(CrossRatioLossBig, self).__init__()
        self.loss_type = loss_type
        self.include_geo = include_geo
        self.geo_loss_gamma_vert = geo_loss_gamma_vert
        self.geo_loss_gamma_horz = geo_loss_gamma_horz
        print(f"Including geometric loss: {include_geo}")
        print(f"Loss type: {loss_type}")

    # input is the heatmap output by the model
    # points is the x,y locations of the points output by the model
    def forward(self, heatmap, points, target_hm, target_points):
        if(self.loss_type == 'l2_softargmax' or self.loss_type == 'l2_sm'):
            mse_loss = (points - target_points) ** 2
            location_loss = mse_loss.sum(2).sum(1).mean()
        elif(self.loss_type == 'l2_heatmap' or self.loss_type == 'l2_hm'):
            mse_loss = (heatmap - target_hm) ** 2
            location_loss = mse_loss.sum(3).sum(2).sum(1).mean()
        elif(self.loss_type == 'l1_softargmax' or self.loss_type == 'l1_sm'):
            l1_loss = torch.abs(points - target_points)
            location_loss = l1_loss.sum(2).sum(1).mean()
        else:
            print("Did not recognize loss function selection!")
            sys.exit(1)

        if self.include_geo:

            # Loss on co-linearity of points along side of cone
            ################# 11 points loss ####################

            v97 = F.normalize(points[:, 9] - points[:, 7], dim=1)
            v75 = F.normalize(points[:, 7] - points[:, 5], dim=1)
            vAA = 1.0 - torch.tensordot(v75, v97, dims=([1], [1]))
            v53 = F.normalize(points[:, 5] - points[:, 3], dim=1)
            vBB = 1.0 - torch.tensordot(v53, v75, dims=([1], [1]))
            v31 = F.normalize(points[:, 3] - points[:, 1], dim=1)
            vCC = 1.0 - torch.tensordot(v31, v53, dims=([1], [1]))
            v10 = F.normalize(points[:, 1] - points[:, 0], dim=1)
            vDD = 1.0 - torch.tensordot(v10, v31, dims=([1], [1]))

            v108 = F.normalize(points[:, 10] - points[:, 8], dim=1)
            v86 = F.normalize(points[:, 8] - points[:, 6], dim=1)
            vEE = 1.0 - torch.tensordot(v86, v108, dims=([1], [1]))
            v64 = F.normalize(points[:, 6] - points[:, 4], dim=1)
            vFF = 1.0 - torch.tensordot(v53, v75, dims=([1], [1]))
            v42 = F.normalize(points[:, 4] - points[:, 2], dim=1)
            vGG = 1.0 - torch.tensordot(v64, v42, dims=([1], [1]))
            v20 = F.normalize(points[:, 2] - points[:, 0], dim=1)
            vHH = 1.0 - torch.tensordot(v42, v20, dims=([1], [1]))

            # Loss on horizontals on cones (color boundaries)
            h21 = F.normalize(points[:, 2] - points[:, 1], dim=1)
            h43 = F.normalize(points[:, 4] - points[:, 3], dim=1)
            hA = 1.0 - torch.tensordot(h43, h21, dims=([1], [1]))
            h65 = F.normalize(points[:, 6] - points[:, 5], dim=1)
            hB = 1.0 - torch.tensordot(h65, h43, dims=([1], [1]))

            h87 = F.normalize(points[:, 8] - points[:, 7], dim=1)
            hC = 1.0 - torch.tensordot(h87, h65, dims=([1], [1]))
            h109 = F.normalize(points[:, 10] - points[:, 9], dim=1)
            hD = 1.0 - torch.tensordot(h109, h87, dims=([1], [1]))

            geo_loss = self.geo_loss_gamma_horz * (hA + hB + hC + hD).mean() / 4 + self.geo_loss_gamma_vert * (vAA + vBB + vCC + vDD + vEE + vFF + vGG + vHH).mean() / 8
            #####################################################

        else:
            geo_loss = torch.tensor(0)
        
        return location_loss, geo_loss, location_loss+geo_loss

