import argparse
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
# from utils.utils import plot_img_and_mask


def predict_img(net,
                full_img,
                device,
                scale_factor=0.5,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        # tf = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((full_img.size[1], full_img.size[0])),
        #     transforms.ToTensor()
        # ])
        #
        # full_mask = tf(probs.cpu()).squeeze()
        # 通道数大于4 不能用他的transform转为PIL图，直接squeeze 2022.04.29
        full_mask = probs.cpu().squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        # return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()  # original
        return full_mask.argmax(dim=0).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    # for debug
    args.model = "./checkpoints/checkpoint_epoch3.pth"
    # args.input = ["./data/test_imgs/02bd959c-f20a2437_0.png", "./data/test_imgs/02c2a4d8-94d32b2f_20.png"]
    save_dir = "./predict/"

    test_dir = '/media/tsy/F/BDD100K/bdd100k_videos_train_00/imgs&labels_1_1/test_image'
    test_mask_dir = '/media/tsy/F/BDD100K/bdd100k_videos_train_00/imgs&labels_1_1/test_label'
    test_list = os.listdir(test_dir)
    args.input = [os.path.join(test_dir, file) for file in test_list]

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=128, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    # summary(net, (3, 360, 640))
    # for layer, param in net.state_dict().items():  # param is weight or bias(Tensor)
    #     print(layer, param.shape, sep=", ")
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')
    count = 0
    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img_ = Image.open(filename)

        mask_ = predict_img(net=net,
                            full_img=img_,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)

        mask_ = mask_ * 2  # 0-128转为0-255  numpy
        # print(filename.split('/')[-1]) # 取文件名
        # cv2.imwrite(os.path.join(save_dir, filename.split('/')[-1]), mask_)

        img_ = torch.from_numpy(BasicDataset.preprocess(img_, 0.5, is_mask=False)).squeeze() # tensor
        mask_path = os.path.join(test_mask_dir, filename.split('/')[-1])
        true_mask = Image.open(mask_path)
        true_mask = torch.from_numpy(BasicDataset.preprocess(true_mask, 0.5, is_mask=True))  # tensor
        sum_abs_error = np.sum(abs(mask_ - true_mask.numpy())) / (360 * 640)
        # sum_error = np.sum(mask_ - true_mask.numpy())
        with open(os.path.join(save_dir, "error.txt"), 'a+') as f:
            f.write(f"{filename.split('/')[-1]} - sum_abs_error : {sum_abs_error}\n")
        print(count)
        count += 1
        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask_)
        #     result.save(out_filename)
        #     logging.info(f'Mask saved to {out_filename}')
        #
        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img_, mask_)
