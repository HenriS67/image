import math
import matplotlib.pyplot as plt
import numpy as np
import imageio
from numpy import linalg as LA
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

class Image:
    def __init__(self, img):
        n=len(img)
        p=len(img[0])
        self.pixels=[[Pixel(0,0,0) for j in range (p)] for i in range(n)]
        for i in range(n):
            for j in range(p):
                px = img[i][j]
                if(type(px)==type(1)):
                    self.pixels[i][j]=Pixel(px,px,px)
                else:
                    r= px[0]
                    g= px[1]
                    b= px[2]
                    self.pixels[i][j]=Pixel(r,g,b)

        self.height=len(self.pixels)
        self.width=len(self.pixels[0])

        self.grey = [[ 0 for j in range (self.width)] for i in range(self.height)]
        for i in range (self.height):
            for j in range(self.width):
                self.grey[i][j]=(self.pixels[i][j].r + self.pixels[i][j].g + self.pixels[i][j].b)/3

    def __str__(self):
        str = ""
    def hist(self):
        hist=np.zeros(256)
        for i in range (self.height):
            for j in range(self.width):
                hist[int(self.grey[i][j])]+=1
        return hist

    def toImg(self):
        img=[[[0,0,0] for j in range (self.width)] for i in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                img[i][j][0] = self.pixels[i][j].r
                img[i][j][1] = self.pixels[i][j].g
                img[i][j][2] = self.pixels[i][j].b
        return img

    def superBorder(self,Img):
        img=self.toImg()

        for i in range(min(Img.height,self.height)):
            for j in range(min(Img.width,self.width)):
                if(Img.pixels[i][j].r!=0 and Img.pixels[i][j].g!=0 and Img.pixels[i][j].b!=0):
                    img[i][j][0] = Img.pixels[i][j].r
                    img[i][j][1] = Img.pixels[i][j].g
                    img[i][j][2] = Img.pixels[i][j].b

        #print(img)
        newImg=Image(img)      
        return newImg
#normalization
def normalize0255(img):
    min=np.min(img)
    max=np.max(img)
    for i in range (len(img)):
        for j in range(len(img[0])):
            img[i][j] = int((img[i][j]/(max-min))*255)
    return img

#lissage
def lissage(Img):
    P=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    for i in range (2,Img.height-2):
        for j in range(2,Img.width-2):
            A = np.array(
                [
                 [Img.grey[i-2][j-2],Img.grey[i-2][j-1],Img.grey[i-2][j],Img.grey[i-2][j+1],Img.grey[i-2][j+2]],
                 [Img.grey[i-1][j-2],Img.grey[i-1][j-1],Img.grey[i-1][j],Img.grey[i-1][j+1],Img.grey[i-1][j+2]],
                 [Img.grey[i-0][j-2],Img.grey[i-0][j-1],Img.grey[i-0][j],Img.grey[i-0][j+1],Img.grey[i-0][j+2]],
                 [Img.grey[i+1][j-2],Img.grey[i+1][j-1],Img.grey[i+1][j],Img.grey[i+1][j+1],Img.grey[i+1][j+2]],
                 [Img.grey[i+2][j-2],Img.grey[i+2][j-1],Img.grey[i+2][j],Img.grey[i+2][j+1],Img.grey[i+2][j+2]]
                 ]) 

            h = (1/159)*np.array([
                [2,4,5,4,2],
                [4,9,12,9,4],
                [5,12,15,12,5],
                [4,9,12,9,4],
                [2,4,5,4,2]
                ])

            Y = A.dot(h)

            P[i][j] = LA.norm(Y)
    return P    
#contour
def contour(Img):
    P=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            P[i][j] = abs(Img.grey[i+1][j] - Img.grey[i-1][j]) + abs(Img.grey[i][j+1] - Img.grey[i][j-1])
    return P

def contourPrewitt(Img):
    P=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            A = np.array(
                [
                 [Img.grey[i-1][j-1],Img.grey[i-1][j],Img.grey[i-1][j+1]],
                 [Img.grey[i][j-1],Img.grey[i][j],Img.grey[i][j+1]],
                 [Img.grey[i+1][j-1],Img.grey[i+1][j],Img.grey[i+1][j+1]]
                 ]) 

            Kgx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
            Kgy = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

            Gx = Kgx.dot(A)
            Gy = Kgy.dot(A)

            P[i][j] = LA.norm(Gx.dot(Gx) + Gy.dot(Gy))
    return P

def contourSobel(Img):
    P=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            A = np.array(
                [
                 [Img.grey[i-1][j-1],Img.grey[i-1][j],Img.grey[i-1][j+1]],
                 [Img.grey[i][j-1],Img.grey[i][j],Img.grey[i][j+1]],
                 [Img.grey[i+1][j-1],Img.grey[i+1][j],Img.grey[i+1][j+1]]
                 ]) 

            Kgx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            Kgy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

            Gx = Kgx.dot(A)
            Gy = Kgy.dot(A)

            P[i][j] = LA.norm(Gx.dot(Gx) + Gy.dot(Gy))
    return P

#voir s'il y a un trou pour ne pas supprimer en trop (pb des coins)
def hole(matrix33):
    res=0
    for i in range (3):
      for j in range(3):
        if(matrix33[i][j]!= 0 and i!=1 and j!=1):
            res+=1
    return res<=2

def contourCanny(Img):

    #contours calcul gradient
    P=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    theta=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            A = np.array(
                [
                 [Img.grey[i-1][j-1],Img.grey[i-1][j],Img.grey[i-1][j+1]],
                 [Img.grey[i][j-1],Img.grey[i][j],Img.grey[i][j+1]],
                 [Img.grey[i+1][j-1],Img.grey[i+1][j],Img.grey[i+1][j+1]]
                 ]) 

            Kgx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            Kgy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

            Gx = Kgx.dot(A)
            Gy = Kgy.dot(A)

            P[i][j] = LA.norm(Gx.dot(Gx) + Gy.dot(Gy))
            if(LA.norm(Gx)!=0):
                theta[i][j] = np.arctan(LA.norm(Gy)/LA.norm(Gx))
    P=normalize0255(P)
    """
    y=477
    x=576
    print("x,y ",x,y," theta: ",P[y][x]," : ",theta[y][x])
    print("x,y ",x-1,y," theta: ",P[y][x-1]," : ",theta[y][x-1])
    """
    #retirer les non-maxima pour avoir un contour unique
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            if P[i][j]!=0:
                cos = math.cos(theta[i][j])
                sin = math.sin(theta[i][j])
                g1 = P[i+round(sin)][j+round(cos)]
                g2 = P[i-round(sin)][j-round(cos)]
                #test
                """
                if(i==477 and j==576):
                    print("g1(x,y) = g1(" , i+math.floor(sin) , ",",j+math.floor(cos),") = ",g1 )
                    print("g2(x,y) = g2(" , i-math.floor(sin) , ",",j-math.floor(cos),") = ",g2 )
                """

                #no holes but can be multiple line
                if ((P[i][j]<=g1) or (P[i][j]<=g2)) and not(sin!=0 and cos!=0):
                    #neighbour matrix
                    """
                    A = np.array(
                        [
                        [P[i-1][j-1],P[i-1][j],P[i-1][j+1]],
                        [P[i][j-1],P[i][j],P[i][j+1]],
                        [P[i+1][j-1],P[i+1][j],P[i+1][j+1]]
                        ]) 
                    if(not hole(A)):
                        """
                    P[i][j] = 0
                
                #holes but one line
                """ 
                if (P[i][j]<g1) or (P[i][j]<g2):
                    
                    P[i][j] = 0
                """   

    #seuillage (binarization)
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1): 
            if  P[i][j]>90:
                P[i][j]=255
            else:
                P[i][j]=0
    return P


#binarisation
def binarisation(Img):
    P=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            if(Img.grey[i][j]!=255):
                P[i][j]=0
            else:
                P[i][j]=255
    return P

def binarisationOtsu(Img):
    nbPixels=Img.width*Img.height
    hist=Img.hist()
    varianceIntraClasse=np.zeros(256)
    #proba de chaque niveau de gris
    #print(hist)
    #print(sum(hist))
    proba=hist/nbPixels
    for i in range(1,256):

        #proba des classes
        proba1=proba[0:i+1]
        #print("i ",i," proba1 :",proba1)
        proba2=proba[i+1:256]
        P1=np.sum(proba1)
        P2=np.sum(proba2)
        
        #moy des classes
        n1=np.arange(i+1)
        #print("n1 :",n1)
        n2=np.arange(i+1,256)

        moy1=sum(n1*proba1)/P1
        moy2=sum(n2*proba2)/P2

        #calcul variances des classes
        var1=sum((n1-moy1)*(n1-moy1)*proba1)
        var2=sum((n2-moy2)*(n2-moy2)*proba2)
        varianceIntraClasse[i] = var1 + var2

    valMin=varianceIntraClasse[1]
    indiceMin=1
    for i in range(1,256):   
        if valMin>varianceIntraClasse[i]:
            valMin=varianceIntraClasse[i]
            indiceMin=i

    #print(indiceMin,valMin)
    P=[[ 0 for j in range (Img.width)] for i in range(Img.height)]
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            if(Img.grey[i][j]>indiceMin):
                P[i][j]=255
            else:
                P[i][j]=0
    return P

#main
img=imageio.imread("test.png").tolist()


Img=Image(img)

#print(Img.pixels[Img.height-1][Img.width-1])
  
liss=normalize0255(lissage(Img))
liImg=Image(liss)
mask=binarisationOtsu(liImg)
bwImg=Image(mask)
#border=normalize0255(contourSobel(bwImg))
border2=normalize0255(contourCanny(bwImg))

#ctImg=Image(border)
ctImg2=Image(border2)

#ctImg.pixels[477][576]=Pixel(255,0,0)

fig = plt.figure(figsize=(10, 7))
fig.add_subplot(2, 2, 1)
  
# showing image
end=Img.superBorder(ctImg2)

plt.imshow(img)
plt.axis('off')
plt.title("originale")

# Adds a subplot at the 2nd position
fig.add_subplot(2, 2, 2)
  
# showing image
plt.imshow(liImg.grey)
plt.axis('off')
plt.title("lissage")

# Adds a subplot at the 2nd position
fig.add_subplot(2, 2, 3)
  
# showing image
plt.imshow(bwImg.grey,cmap='gray')
plt.axis('off')
plt.title("binarisation")

# Adds a subplot at the 2nd position
fig.add_subplot(2, 2, 4)
  
# showing image
plt.imshow(ctImg2.grey,cmap='gray')
plt.axis('off')
plt.title("contours")


plt.show()
