import torch
import math
import cmath
import torchvision
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

#Construct Hadamard matrix by Sylvester's construction + combine it with my idea
def hadamard(order):
    if order == 0:
        element = 1
        return [torch.tensor(element)]
    else:
        element_list = hadamard(order-1)
        matrix_list = []
        matrix_list.append(torch.tensor([[1,1],[1,-1]]))
        matrix_list.append(torch.tensor([[-1,1],[1,1]]))
        matrix_list.append(torch.tensor([[1,-1],[1,1]]))
        matrix_list.append(torch.tensor([[1,1],[-1,1]]))
        for n in range(len(matrix_list)):
            matrix_list[n] = torch.repeat_interleave\
                (matrix_list[n],2**(order-1),dim=1)
            matrix_list[n] = torch.repeat_interleave\
                (matrix_list[n],2**(order-1),dim=0)
        result = []
        for i in range(len(element_list)):
            matrix = torch.tile(element_list[i],(2,2))
            for n in range(len(matrix_list)):
                result.append(torch.mul(matrix,matrix_list[n]))
        return result

def main():

    input_mode = 4096
    output_mode = 4096
    ccd_width = int(math.sqrt(output_mode))
    imaginary = torch.randn(input_mode*output_mode)
    real = torch.torch.randn(input_mode*output_mode)
    tm = real+1j*imaginary
    tm = torch.reshape(tm, (output_mode, input_mode))
    print(torch.mean(tm))
    print(torch.var(tm))

    amp = torch.rand(1)
    phase = torch.exp(2j*math.pi*torch.rand(1))
    ref_field = torch.tile(amp*phase, [input_mode, 1])
    #ref_field after TM
    ref_field = torch.matmul(tm,ref_field)

    '''Canonical basis'''
    controlled_phase = (torch.randperm(1000000)[:input_mode]/1000000)
    controlled_phase = torch.exp(2j*math.pi*controlled_phase)
    controlled_phase = torch.unsqueeze(controlled_phase,1)
    input_field = torch.mul(ref_field, controlled_phase)

    #It will be transposed
    observed_tm = torch.zeros(input_mode, output_mode,dtype=torch.complex64)

    for n in tqdm(range(0, input_mode)):
        zero = torch.zeros(input_mode, 1)
        zero[n, 0] = 1
        field = torch.mul(input_field, zero)
        output_field = torch.matmul(tm, field)

        inten = []
        for i in range(0, 4):
            step = cmath.exp(0.5j*i*math.pi)*output_field
            inten_i = torch.square(torch.abs(ref_field + step))
            inten.append(inten_i)

        #Phase stepping
        res = (inten[0]-inten[2])/4 + 1j*(inten[3]-inten[1])/4
        observed_tm[n] = torch.transpose(res,0,1)
        '''
        other = torch.conj(ref_field)
        res = torch.transpose(res,0,1)
        other = torch.transpose(other,0,1)
        res = torch.div(res,other)
        observed_tm[n] = res*(1/field[n])
        '''
    observed_tm = torch.transpose(observed_tm,0,1)
    observed_tm /= torch.norm(observed_tm)

    original_speckle = torch.square(torch.abs(ref_field+\
                                              torch.matmul(tm,input_field)))
    original_speckle = torch.reshape(original_speckle,(ccd_width,ccd_width))
    #original_speckle = original_speckle/torch.max(original_speckle)

    recon_speckle = torch.square(torch.abs(ref_field+\
                                        torch.matmul(observed_tm,input_field)))
    recon_speckle = torch.reshape(recon_speckle,(ccd_width,ccd_width))
    #recon_speckle = recon_speckle/torch.max(recon_speckle)

    transform = T.ToPILImage()
    img_t = transform(torch.abs(recon_speckle))
    plt.imshow(img_t)
    plt.show()
    img_t.save('./recon.png')
    img_t = transform(torch.abs(original_speckle))
    plt.imshow(img_t)
    plt.show()
    img_t.save('./origin.png')

'''
    #Hadamard basis TM
    hadamard_tm = torch.zeros(input_mode, ouptut_mode, dtype=torch.complex128)
    hadamard_matrix_list = hadamard(math.log(math.sqrt(input_mode),2))
    for n in tqdm(range(0,input_mode)):
        hadamard_matrix =
'''



if __name__=="__main__":
    main()
