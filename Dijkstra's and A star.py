'''Dijkstra's Algorithm'''

import cv2

map=cv2.imread("processedmaze.png")

r=(255,0,0)
b=(0,0,255)
w=(255,255,255)
yell=(0,255,255)
bl=(0,0,0)
v=(255,0,255)
start=(335,139)
end=(226,720)

map[start]=r
map[end]=b

set={}
count=0

'''To store the information of the map in a dictionary with elements {(coordinate):[dist. from the start node(g), parent node, flag to indicate if it is visited]}'''
for i in range(414):
    for j in range(747):
        if (map[i][j]==w).all() or (map[i][j]==r).all() or (map[i][j]==b).all() :
            if ((i,j)==start):
                set[(i,j)]=[0,[],0]
            else:
                set[(i,j)]=[-1,[],0]
            count+=1
print(count)

def minset():
    '''Returns the coordinate having minimum value of g'''
    f=0 #To indicate whether minsi is assigned a value in the function
    for k in set:
        if (set[k][2]==0):
            if (f==0 and set[k][0]>=0):
                f=1
                mins=set[k][0]
                minsi=k
            elif (f==1 and set[k][0]<=mins and set[k][0]>=0):
                mins=set[k][0]
                minsi=k
    #print(mins, minsi)
    if (f==1):
        return minsi
    else:
        return None
while (count):
    k = minset()
    if (k!=None):
        for (i,j) in [(1,0),(0,1),(-1,0),(0,-1)]: #To traverse through all the neighbours
            if ((k[0]+i,k[1]+j) in set):
                if (set[(k[0]+i,k[1]+j)][2]==0 and set[(k[0]+i,k[1]+j)][0]==-1):
                    set[(k[0]+i,k[1]+j)][0]=set[k][0]+1
                    set[(k[0]+i,k[1]+j)][1]=k
                else:
                    if (set[(k[0]+i,k[1]+j)][0]>(set[k][0]+1)):
                        set[(k[0]+i,k[1]+j)][0]=set[k][0]+1
                        set[(k[0]+i,k[1]+j)][1]=k
        if (k==end):
            break
        set[k][2]=1
        map[k[0]][k[1]]=yell
        cv2.imshow("Map",map)
        cv2.waitKey(1);
        count-=1
    if (k==None):
        break

'''To store the pixels in the shortest path'''

slist=[]
n=end

while (n!=start):
    slist.append(n)
    n=set[n][1]
for i in range(len(slist)):
    '''Coloring the shortest from path from start node to end node'''
    map[slist[len(slist)-i-1][0]][slist[len(slist)-i-1][1]]=v
    cv2.imshow("Map",map)
    cv2.waitKey(10)
cv2.imshow("Map",map)
cv2.waitKey(0)
cv2.destroyAllWindows


'''A* algorithm'''

import cv2
import math

map=cv2.imread("processedmaze.png")

r=(255,0,0)
b=(0,0,255)
w=(255,255,255)
yell=(0,255,255)
bl=(0,0,0)
v=(255,0,255)
start=(335,139)
end=(226,720)

def heuristic(i,j):
    s=math.sqrt((end[0]-i)**2 + (end[1]-j)**2)
    return s

def minset():
    '''Returns the coordinate having minimum value of g+h'''
    mini=0
    r=()
    flag=0 #To indicate whether r is assigned a value in the function
    for i in dict:
        if (dict[i][2]==0):
            if (flag==0 and dict[i][4]>=0):
                mini=dict[i][4]
                r=i
                flag=1
            elif(flag==1 and dict[i][4]>=0):
                if (mini>dict[i][4]):
                    mini=dict[i][4]
                    r=i
    if (r!=()):
        return r
    else:
        return None
map[start]=r
map[end]=b
dict={}
count=0

'''To store the information of the map in a dictionary with elements {(coordinate):[dist. from the start node(g), parent node, flag to indicate if it is visited, heuristic value(h), g+h]}'''
for i in range(414):
    for j in range(747):
        if (map[i][j]==w).all() or (map[i][j]==r).all() or (map[i][j]==b).all():
            if (i==start[0] and j==start[1]):
                dict[(i,j)]=[0,(),0,heuristic(i,j),heuristic(i,j)]
            else:
                dict[(i,j)]=[-1,(),0,heuristic(i,j),-1]
            count+=1


while(count):
    k=minset()
    if (k==None):
        break
    for (i,j) in [(1,0),(-1,0),(0,1),(0,-1)]: #To traverse through all the neighbours
        if ((k[0]+i,k[1]+j) in dict):
            if (dict[(k[0]+i,k[1]+j)][0]<0):
                dict[(k[0] + i, k[1] + j)][0] = (dict[k][0] + 1)
                dict[(k[0] + i, k[1] + j)][1] = k
                dict[(k[0] + i, k[1] + j)][4] = dict[(k[0] + i, k[1] + j)][0] + dict[(k[0] + i, k[1] + j)][3]
            elif (dict[(k[0]+i,k[1]+j)][0]>=0 and dict[(k[0]+i,k[1]+j)][0]>(dict[k][0]+1)):
                dict[(k[0]+i,k[1]+j)][0]=(dict[k][0]+1)
                dict[(k[0]+i,k[1]+j)][1]=k
                dict[(k[0]+i,k[1]+j)][4]=dict[(k[0]+i,k[1]+j)][0]+dict[(k[0]+i,k[1]+j)][3]
    if (k==end):
        break
    map[k] = yell
    dict[k][2] = 1
    cv2.imshow("Map",map)
    cv2.waitKey(5)

finalpath=[]
n=end

'''To store the pixels in the shortest path'''
while (n!=start):
    finalpath.append(n)
    n=dict[n][1]

for i in range(len(finalpath)):
    '''Coloring the shortest from path from start node to end node'''
    map[finalpath[len(finalpath)-i-1][0]][finalpath[len(finalpath)-i-1][1]]=v
    cv2.imshow("Map",map)
    cv2.waitKey(5)


cv2.imshow("Map",map)
cv2.waitKey(0)
cv2.destroyAllWindows()

