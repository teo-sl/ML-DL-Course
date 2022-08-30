import numpy as np

def convolution(input,filter):
    output_height=input.shape[0]-filter.shape[0]+1
    output_width=input.shape[1]-filter.shape[1]+1
    output_image=np.zeros((output_height,output_width))
    for i in range(0,output_height):
        for j in range(0,output_width):
            for m in range(0,filter.shape[0]):
                for n in range(0,filter.shape[1]):
                    output_image[i,j]+=input[i+m,j+n]*filter[m,n]
    
    return output_image

a=np.array([[1,2,3,4],[0,1,2,3],[4,5,6,0],[1,3,0,0],[0,0,1,2]])
filter=np.array([[0,1,1],[1,1,1],[1,0,1]])

print(convolution(a,filter))

