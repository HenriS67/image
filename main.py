import math
import random
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
                if(type(px)==type(1) or type(px)==type(1.0)):
                    self.pixels[i][j]=Pixel(int(px),int(px),int(px))
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
                    img[i][j] = [255,0,0]
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

            if(LA.norm(Gx)!=0):
                theta[i][j] = np.arctan(LA.norm(Gy)/LA.norm(Gx))

            P[i][j] = LA.norm(Gx.dot(Gx) + Gy.dot(Gy))

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

    """
    #ajout extreme de l'image comme contour
    for i in range (Img.height):
        P[i][0]
        P[i][Img.width-1]
    for j in range(Img.width): 
        P[0][j]
        P[Img.height-1][j]
"""
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

#decalage de 1 px pour les parties collées aux extremités
def recadrage(Img):
    #Traitement Image : ajout pixel à l'extreme dezl'image comme contour

    P=[[ 0 for j in range (Img.width+2)] for i in range(Img.height+2)]

    #contour blanc
    for i in range (Img.height+2):
        P[i][0]=255
        P[i][Img.width+2-1]=255
    for j in range(Img.width+2): 
        P[0][j]=255
        P[Img.height+2-1][j]=255

    for i in range (1,Img.height+2-1):
        for j in range(1,Img.width+2-1):
            P[i][j]=int(Img.grey[i-1][j-1])

    #print(P)
    return P

def contourBlanc(Img):
    
    resImg=Img
    #contour blanc
    noirOuBlanc = round((np.average(resImg.grey)+30)/255)*255
    print(np.average(resImg.grey))
    for i in range (Img.height):
        resImg.grey[i][0]=noirOuBlanc
        resImg.grey[i][Img.width-1]=noirOuBlanc
        resImg.grey[i][1]=noirOuBlanc
        resImg.grey[i][Img.width-2]=noirOuBlanc
    for j in range(Img.width): 
        resImg.grey[0][j]=noirOuBlanc
        resImg.grey[Img.height-1][j]=noirOuBlanc
        resImg.grey[1][j]=noirOuBlanc
        resImg.grey[Img.height-2][j]=noirOuBlanc

    return resImg.grey
#contours -> rectangulization (parcours en profondeur itératif)
from collections import deque
def rectangulization(Img):

    P=[[ -1 for j in range (Img.width)] for i in range(Img.height)]
    nbPart=-1

    #partition des contours
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            if(Img.grey[i][j]!=0 and P[i][j]==-1):
                nbPart+=1
                node=(i,j)
                P[i][j]=nbPart
                visited = []
                stack = deque()
                stack.append(node)
                while stack:
                    node = stack.pop()
                    iA=node[0]
                    jA=node[1]
                    P[iA][jA]=nbPart
                    if node not in visited:
                        visited.append(node)
                        unvisited = []
                        if(Img.grey[iA-1][jA-1]!=0 and (iA-1,jA-1) not in visited):unvisited.append((iA-1,jA-1))
                        if(Img.grey[iA-1][jA]!=0 and (iA-1,jA) not in visited):unvisited.append((iA-1,jA))
                        if(Img.grey[iA-1][jA+1]!=0 and (iA-1,jA+1) not in visited):unvisited.append((iA-1,jA+1))
                        if(Img.grey[iA][jA-1]!=0 and (iA,jA-1) not in visited):unvisited.append((iA,jA-1))
                        if(Img.grey[iA][jA]!=0 and (iA,jA) not in visited):unvisited.append((iA,jA))
                        if(Img.grey[iA][jA+1]!=0 and (iA,jA+1) not in visited):unvisited.append((iA,jA+1))
                        if(Img.grey[iA+1][jA-1]!=0 and (iA+1,jA-1) not in visited):unvisited.append((iA+1,jA-1))
                        if(Img.grey[iA+1][jA]!=0 and (iA+1,jA) not in visited):unvisited.append((iA+1,jA))
                        if(Img.grey[iA+1][jA+1]!=0 and (iA+1,jA+1) not in visited):unvisited.append((iA+1,jA+1))

                        stack.extend(unvisited)

    #création carre[(x1,y1,x2,y2)] contours des formes
    carres=[ [Img.width,Img.height,0,0] for j in range (nbPart+1)]
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            if(P[i][j]!=-1):

                if(i<carres[P[i][j]][1]):
                    carres[P[i][j]][1]=i
                if(i>carres[P[i][j]][3]):
                    carres[P[i][j]][3]=i 
                if(j<carres[P[i][j]][0]):
                    carres[P[i][j]][0]=j
                if(j>carres[P[i][j]][2]):
                    carres[P[i][j]][2]=j 

    #enleve les carrés imbriqués dans d'autres (opti possible (break))
    tabSimpl=[-1 for i in range(nbPart+1)]
    for i in range(nbPart+1):
        tabSimpl[i]=i
        for j in range(nbPart+1):
            if(carres[i][0]>=carres[j][0] 
                and carres[i][2]<=carres[j][2]
                and carres[i][1]>=carres[j][1]
                and carres[i][3]<=carres[j][3]):
                tabSimpl[i]=j
                break

    #remplissage (on rempli le carré s'il est isolé, on rempli le contour si conflits entre carrés):
    #((X,Y),(A,B)) and ((X1,Y1),(A1,B1)) if (A<X1 or A1<X or B<Y1 or B1<Y):
    # = > Intersection = Empty
    for i in range(nbPart+1):

        if(i in tabSimpl):
            condiI=True
            for j in range(nbPart+1):
                if(j in tabSimpl):
                    if not (carres[i][2]<carres[j][0] or 
                        carres[j][2]<carres[i][0] or 
                        carres[i][3]<carres[j][1] or
                        carres[j][3]<carres[i][1]) and i!=j:
                        condiI = False
                        #print(i,j)
                        break

        #si aucune intersection (rectangle isolé), forme = rectangle
        if condiI:
            for m in range(carres[i][1],carres[i][3]):
                for l in range(carres[i][0],carres[i][2]):
                    P[m][l]=tabSimpl[i]
        
        #sinon on prend le contour (fermé)
        else:
            #méthode appartenance à un contour ( nb intesection)
            """
        
            for m in range(carres[i][1],carres[i][3]+1):
                nbCrois=0
                tabIn=[]
                for l in range(carres[i][0],carres[i][2]+1):
                    if(P[m][l]==tabSimpl[i] and P[m][l+1]==-1):
                        nbCrois+=1
                        if(nbCrois%2==0):
                            for o in range(len(tabIn)):
                                P[tabIn[o][0]][tabIn[o][1]]=tabSimpl[i]
                            tabIn=[]
                    if (not (nbCrois%2==0)):
                        tabIn.append((m,l))
                    if(m,l)==(38,501):
                        print(nbCrois)
                        print(P[m][l-1],P[m][l],P[m][l+1])
                        print(tabIn)
            """   
            #méthode min-max par ligne (plus large)
        for m in range(carres[i][1],carres[i][3]+1):
                min=carres[i][2]+1
                max=carres[i][0]
                for l in range(carres[i][0],carres[i][2]+1):
                    if(P[m][l]==tabSimpl[i]):
                        if(l>max):
                            max=l
                        if(l<min):
                            min=l
                for l in range(min,max):
                    P[m][l]=   tabSimpl[i]               

    res=[[ [0,0,0] for j in range (Img.width)] for i in range(Img.height)]
    tabColor=[[random.randint(0,255),random.randint(0,255),random.randint(0,255)] for i in range(nbPart+1)]

    #dessins des carrés
    
    for i in range (1,Img.height-1):
        for j in range(1,Img.width-1):
            if(P[i][j]!=-1 and P[i][j]==tabSimpl[P[i][j]]):
                res[i][j]=tabColor[tabSimpl[P[i][j]]]
    
    """
    for i in range(nbPart+1):
        if(i in tabSimpl):
            for m in range(carres[i][1],carres[i][3]):
                res[m][carres[i][0]]=[255,0,0]
                res[m][carres[i][2]]=[255,0,0]
            for m in range(carres[i][0],carres[i][2]):
                res[carres[i][1]][m]=[255,0,0]
                res[carres[i][3]][m]=[255,0,0]
    """

    #print(nbPart)


    return res
#main
img=imageio.imread("gg.jpg").tolist()


Img=Image(img)

#print(Img.pixels[Img.height-1][Img.width-1])
  
print("lissage")
liss=normalize0255(lissage(Img))
liImg=Image(liss)
print("binarization")
mask=binarisationOtsu(liImg)
bwImg=Image(mask)
#border=normalize0255(contourSobel(bwImg))

print("cadre Blanc")
recadr=contourBlanc(bwImg)
reImg=Image(recadr)

print("Contour")
border2=normalize0255(contourCanny(reImg))

#ctImg=Image(border)
ctImg2=Image(border2)

#ctImg.pixels[477][576]=Pixel(255,0,0)
print("Rectangulization")
imgend=rectangulization(ctImg2)
fig = plt.figure(figsize=(10, 7))
fig.add_subplot(1, 2, 1)
  
# showing image
##end=Img.superBorder(ctImg2)

plt.imshow(img)
fig.add_subplot(1, 2, 2)
plt.imshow(imgend)


fig = plt.figure(figsize=(10, 7))
fig.add_subplot(3, 3, 1)
  
# showing image
end=Img.superBorder(ctImg2)

plt.imshow(end.toImg())
plt.axis('off')
plt.title("originale")

# Adds a subplot at the 2nd position
fig.add_subplot(3, 3, 2)
  
# showing image
plt.imshow(liImg.grey)
plt.axis('off')
plt.title("lissage")

# Adds a subplot at the 2nd position
fig.add_subplot(3, 3, 3)
  
# showing image
plt.imshow(bwImg.grey,cmap='gray')
plt.axis('off')
plt.title("binarisation")

# Adds a subplot at the 2nd position
fig.add_subplot(3, 3, 4)
  
# showing image
plt.imshow(ctImg2.grey,cmap='gray')
plt.axis('off')
plt.title("contours")


fig.add_subplot(3, 3, 5)
  
# showing image
plt.imshow(imgend)
plt.axis('off')
plt.title("parts")

plt.show()
