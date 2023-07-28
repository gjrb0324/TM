import torch
import torchvision.transforms as T
import math
def main():
    input_mode = 10000
    output_mode = 10000
    ccd_width = int(math.sqrt(output_mode))
    theta = 2j*math.pi*torch.rand(input_mode*output_mode)
    phase = torch.exp(theta)
    t = torch.rand(input_mode*output_mode)
    tm = torch.mul(t, phase)
    tm = torch.reshape(tm, (output_mode, input_mode))
    res = torch.sum(tm,1)
    res = torch.reshape(res, (100,100))

    transform = T.ToPILImage()
    img_t = transform(torch.abs(res))
    img_t.show()

if __name__ == "__main__":
    main()
