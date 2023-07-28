import torch
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


hadamard(3)
