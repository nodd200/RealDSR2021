import torch
import torch.nn.functional as F
import torch.nn as nn
import octconv as oc

def grid_generator(k, r, n):
    """grid_generator
    Parameters
    ---------
    f : filter_size, int
    k: kernel_size, int
    n: number of grid, int
    Returns
    -------
    torch.Tensor. shape = (n, 2, k, k)
    """
    grid_x, grid_y = torch.meshgrid([torch.linspace(k//2, k//2+r-1, steps=r),
                                     torch.linspace(k//2, k//2+r-1, steps=r)])
    grid = torch.stack([grid_x,grid_y],2).view(r,r,2)

    return grid.unsqueeze(0).repeat(n,1,1,1).cuda()


    
class MS_RB(nn.Module):
    def __init__(self, num_feats, kernel_size):
        super(MS_RB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=kernel_size, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x3 = x1 + x2
        x4 = self.conv4(x3)
        out = x4 + x

        return out

def resample_data(input, s):
    """
        input: torch.floatTensor (N, C, H, W)
        s: int (resample factor)
    """    
    
    assert( not input.size(2)%s and not input.size(3)%s)  #判断H，W能否被s整除

    
    if input.size(1) == 3:   #RGB 3通道
        
        # bgr2gray (same as opencv conversion matrix)
        input = (0.299 * input[:,2] + 0.587 * input[:,1] + 0.114 * input[:,0]).unsqueeze(1)
        
    out = torch.cat([input[:,:,i::s,j::s] for i in range(s) for j in range(s)], dim=1)

    """
        out: torch.floatTensor (N, s**2, H/s, W/s)
    """
    return out

class Net(nn.Module):
    def __init__(self, num_feats, depth_chanels, color_channel, kernel_size,scale = 4):
        super(Net, self).__init__()
        self.filter_size = 15
        self.kernel_size = kernel_size
        self.conv_rgb1 = nn.Conv2d(in_channels=16, out_channels=num_feats, 
                                   kernel_size=kernel_size, padding=1)
        #self.conv_rgb2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        #self.conv_rgb3 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        #self.conv_rgb4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        #self.conv_rgb5 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
                                   
        self.rgb_cbl2 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0, alpha_out=0.25,
                                    stride=1, padding=1,dilation=1,groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.rgb_cbl3 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.25, alpha_out=0.25,
                                    stride=1, padding=1,dilation=1,groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))   
        self.rgb_cbl4 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.25, alpha_out=0.25,
                                    stride=1, padding=1,dilation=1,groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        #self.rgb_cbl5 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.125, alpha_out=0.125,
        #                            stride=1, padding=1,dilation=1,groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))                         
                                   
                                   
    


        self.conv_dp1 = nn.Conv2d(in_channels=16, out_channels=num_feats, 
                                  kernel_size=kernel_size, padding=1)
        self.MSB1 = MS_RB(num_feats, kernel_size)
        self.MSB2 = MS_RB(56, kernel_size)
        self.MSB3 = MS_RB(80, kernel_size)
        self.MSB4 = MS_RB(104, kernel_size)
        self.conv1 = nn.Conv2d(104,128,3,padding= 1)
        self.conv2 = nn.Conv2d(24,128,3,padding= 1)
        self.conv_weight = nn.Conv2d(128, kernel_size**2*scale**2, 1)
        self.conv_offset = nn.Conv2d(128, 2*kernel_size**2*(scale)**2, 1)

        self.conv_recon1 = nn.Conv2d(in_channels=104, out_channels=4 * num_feats, kernel_size=kernel_size, padding=1)
        self.ps1 = nn.PixelShuffle(2)
        self.conv_recon2=nn.Conv2d(in_channels=num_feats, out_channels=4 * num_feats, kernel_size=kernel_size, padding=1)
        self.ps2 = nn.PixelShuffle(2)
        self.restore=nn.Conv2d(in_channels=num_feats, out_channels=1, kernel_size=kernel_size, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self,  x):
        image, depth = x

        re_im = resample_data(image, 4)
        re_dp = resample_data(depth, 4)
        
        dp_in = self.act(self.conv_dp1(re_dp))
        dp1 = self.MSB1(dp_in)

        rgb1 = self.act(self.conv_rgb1(re_im))
        #rgb2 = self.act(self.conv_rgb2(rgb1))
        
        rgb2 = self.rgb_cbl2(rgb1)
        
        ca1_in = torch.cat([dp1,rgb2[0]],dim = 1)
        dp2 = self.MSB2(ca1_in)
        #rgb3 = self.conv_rgb3(rgb2)
        rgb3 = self.rgb_cbl3(rgb2)
        #ca2_in = dp2 + rgb3
        ca2_in = torch.cat([dp2,rgb3[0]],dim = 1)

        dp3 = self.MSB3(ca2_in)
        #rgb4 = self.conv_rgb4(rgb3)
        rgb4 = self.rgb_cbl4(rgb3)

        #ca3_in = rgb4 + dp3
        ca3_in = torch.cat([dp3,rgb4[0]],dim = 1)
        
        dp4 = self.MSB4(ca3_in)

        x = self.conv1(dp4)

        weight = torch.sigmoid(self.conv_weight(x))
        offset = torch.sigmoid(self.conv_offset(x))
        


        ps = nn.PixelShuffle(4)
        weight = ps(weight)
        offset = ps(offset)


        
        weight -= torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        
        b, h, w = image.size(0), image.size(2), image.size(3)
        k = self.filter_size
        r = self.kernel_size
        hw = h*w

        weight = weight.permute(0,2,3,1).contiguous().view(b*hw, r*r, 1)
        offset = offset.permute(0,2,3,1).contiguous().view(b*hw, r,r, 2)
        
        grid = grid_generator(k, r, b*hw)
        coord = grid + offset
        coord = (coord / k * 2) -1
        
        
        
        depth_col = F.unfold(depth, k, padding=k//2).permute(0,2,1).contiguous().view(b*hw, 1, k,k)
        depth_sampled = F.grid_sample(depth_col,coord).view(b*hw, 1, 9)
        out = torch.bmm(depth_sampled, weight).view(b, 1, h,w)
        #up1 = self.ps1(self.conv_recon1(self.act(dp4)))
        #up2 = self.ps2(self.conv_recon2(up1))
        #out = self.restore(up2)
        out = depth + out

        return out