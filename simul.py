import torch
import math
import cmath
import torchvision
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import random

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

def quarter_circle_law(mode_num):
    """
    mode_num:min(output_mode_num,input_mode_num)
        which means the number of singular values
    """
    num = 0
    singular_list = torch.zeros(mode_num,dtype=torch.complex64)

    while num < mode_num:
        x = 2*random.random()
        y = random.random()
        res = math.sqrt(4-x*x)/math.pi
        if y <= res:
            singular_list[num]=x
        num +=1

    return singular_list

def main():

    input_mode = 256
    output_mode = 256
    ccd_width = int(math.sqrt(output_mode))
    theta = torch.rand(input_mode*output_mode)
    tm = torch.exp(2j*math.pi*theta)
    #imaginary = torch.randn(input_mode*output_mode)
    #real = torch.torch.randn(input_mode*output_mode)
    #tm = real+1j*imaginary
    tm = torch.reshape(tm, (output_mode, input_mode))
    #tm /= math.sqrt(output_mode)

    #U,S,V = torch.linalg.svd(tm)
    #zero_temp = torch.zeros((output_mode,input_mode),dtype=torch.complex64)
    #singular_list = quarter_circle_law(min(output_mode,input_mode))
    #for i in range(min(output_mode,input_mode)):
    #    zero_temp[i,i]=singular_list[i]

    #S = zero_temp
    #tm = torch.matmul(torch.matmul(U,S), \
    #                  torch.transpose(torch.conj(V),0,1))
    #tm = torch.mul(tm,math.sqrt(output_mode))

    print(torch.mean(tm))
    print(torch.var(tm))

    amp = torch.rand(1)
    phase = torch.exp(2j*math.pi*torch.rand(1))
    ref_field_in = torch.tile(amp*phase, [input_mode, 1])
    #ref_field after TM
    ref_field_out = torch.matmul(tm,ref_field_in)

    '''Canonical basis'''
    controlled_phase = (torch.randperm(1000000)[:input_mode]/1000000)
    controlled_phase = torch.exp(2j*math.pi*controlled_phase)
    controlled_phase = torch.unsqueeze(controlled_phase,1)
    input_field = torch.mul(ref_field_in, controlled_phase)

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
            inten_i = torch.square(torch.abs(ref_field_out + step))
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

    #row normalisation
    filtered_tm_1 = observed_tm/torch.norm(observed_tm,dim=1)
    #column normalisation
    row_norm = torch.unsqueeze(torch.norm(observed_tm,dim=0),0)
    row_norm = torch.transpose(row_norm,0,1)
    filtered_tm_2 = observed_tm/row_norm
    #Total normalisation
    filtered_tm_3 = observed_tm/torch.norm(observed_tm)

    original_speckle = torch.square(torch.abs(ref_field_out+\
                                              torch.matmul(tm,input_field)))
    original_speckle = torch.reshape(original_speckle,(ccd_width,ccd_width))
    #original_speckle = original_speckle/torch.max(original_speckle)

    recon_speckle = torch.square(torch.abs(ref_field_out+\
                                        torch.matmul(observed_tm,input_field)))
    recon_speckle = torch.reshape(recon_speckle,(ccd_width,ccd_width))
    #recon_speckle = recon_speckle/torch.max(recon_speckle)

    transform = T.ToPILImage()
    img_t = transform(torch.abs(recon_speckle))
    #plt.imshow(img_t)
    #plt.show()
    img_t.save('./recon.png')
    img_t = transform(torch.abs(original_speckle))
    #plt.imshow(img_t)
    #plt.show()
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
