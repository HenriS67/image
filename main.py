import matplotlib.pyplot as plt
import numpy as np
import imageio

class Pixel:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
     
    def __str__(self):
        return "({0},{1},{2})".format(self.r, self.g, self.b)
 
    def __sub__(self, p):
        r = self.r - p.r
        g = self.g - p.g
        b = self.b - p.b
        return Pixel(r, g, b)

    def __add__(self, p):
        r = self.r + p.r
        g = self.g + p.g
        b = self.b + p.b
        return Pixel(r, g, b)

    def noColor(self):
        return (r + g + b)/3

class Image:
    def __init__(self, pixels):
        self.pixels = pixels
        self.height=len(pixels)
        self.width=len(pixels[0])

        self.grey = [[ 0 for j in range (self.width)] for i in range(self.height)]
        for i in range (self.height):
            for j in range(self.width):
                self.grey[i][j]=(pixels[i][j].r + pixels[i][j].g + pixels[i][j].b)/3

    def __str__(self):
        str = ""

def contour(Img):
    P=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            P[i][j] = abs(Img.grey[i+1][j] - Img.grey[i-1][j]) + abs(Img.grey[i][j+1] - Img.grey[i][j-1])
    return P

def binarisation(Img):
    P=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            if(Img.grey[i][j]!=255):
                P[i][j]=0
            else:
                P[i][j]=255
    return P

def contourSobel(Img):
    P=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    return P

def contourPrewitt(Img):
    P=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            A = np.array([[1,2,0],[4,3,-1],[4,3,-1]]) 
            Kgx = np.array([[1,2,0],[4,3,-1],[4,3,-1]])
            Kgy = np.array([[1,2,0],[4,3,-1],[4,3,-1]])

            Gx = Kgx.dot(A)
            Gy = Kgy.dot(A)

            P[i][j] = np.norm(Gx.dot(Gx) + Gy.dot(Gy))
    return P

img=imageio.imread("image.jpg").tolist()
n=len(img)
p=len(img[0])
image=[[Pixel(0,0,0) for j in range (p)] for i in range(n)]
for i in range(n):
    for j in range(p):
        r= img[i][j][0]
        g= img[i][j][1]
        b= img[i][j][2]
        image[i][j]=Pixel(r,g,b)

Img=Image(image)

print(Img.pixels[Img.height-1][Img.width-1])
  
border=contourPrewitt(Img)
mask=binarisation(Img)
plt.figure()
plt.imshow(border)
plt.show()
