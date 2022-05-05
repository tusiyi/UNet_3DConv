import torch


if __name__ == "__main__":
    content = torch.load("./checkpoints/checkpoint_epoch9.pth")
    for key, value in content.items():
        print(key, value.shape, sep=" : ")
    # print(content.keys())
