import torch
from packnet_sfm.networks import packnet
from packnet_sfm.networks.layers import InvDepth
import matplotlib.pyplot as plt

import cv2

def main1():
    def onnx_out():
        example_inputs = torch.rand(1, 3, 640, 192).cuda()
        encoder_out = torch.onnx.export(model=arch,
                                        args=example_inputs,
                                        f='./packnet.onnx',
                                        verbose=True,
                                        export_params=True  # 带参数输出
                                        )



    arch = packnet.PackNet01(version='1A', dropout=0.0)
    arch.eval()
    arch.to('cuda')


    loaded_dict = torch.load('/home/roit/models/packnet/PackNet01_MR_selfsup_K.pth', map_location='cuda')
    arch.load_state_dict(loaded_dict)


    # arch2= packnet.PackNet01(version='1A',dropout=0.0)
    # arch2.eval()
    # arch2.to('cuda')
    #
    # loaded_dict = torch.load('/home/roit/models/packnet/PackNet01_MR_selfsup_K.ckpt', map_location='cuda')['state_dict']
    # # Get network state dict
    # network_state_dict = arch.state_dict()
    #
    # updated_state_dict = OrderedDict()
    # n, n_total = 0, len(network_state_dict.keys())
    # for key, val in loaded_dict.items():
    #     for prefix in ['depth_net', 'disp_network'] :
    #         prefix = prefix + '.'
    #         if prefix in key:
    #             idx = key.find(prefix) + len(prefix)
    #             key = key[idx:]
    #             if key in network_state_dict.keys() and \
    #                     same_shape(val.shape, network_state_dict[key].shape):
    #                 updated_state_dict[key] = val
    #                 n += 1
    #
    # arch2.load_state_dict(updated_state_dict, strict=False)


    img_np = cv2.imread('../img.png')
    img_np=cv2.resize(img_np,(640,192))
    img_np = img_np.transpose([2,0,1])
    img = torch.tensor(img_np).to('cuda',dtype=torch.float32)
    img = img.unsqueeze(dim=0)

    disp = arch(img)
    depth = 1. / disp.clamp(min=1e-6)

    depth = depth.detach().cpu().numpy()[0][0]
    plt.imshow(depth,cmap='plasma')
    plt.show()




    print('ok')
def main2():
    network = torch.load('/home/roit/models/packnet/ResNet18_MR_selfsup_K.ckpt')

    print('ko')
    pass

if __name__ == '__main__':
    main1()





