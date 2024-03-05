import math
import random

def ackleyFunction(x,y):
    a = 20
    b = 0.2
    c = 2*math.pi
    partOne = -a*math.exp(-b*math.sqrt(0.5*(x**2+y**2)))
    partTwo = -math.exp(0.5*(math.cos(c*x)+math.cos(c*y)))+ a + math.exp(1)
    ackley = partOne + partTwo
    return ackley


def grid(tekrar,step,x,y):
    a= random.uniform(-x,x)
    b= random.uniform(-y,y)
    bestSolution = ackleyFunction(a,b)
    for i in range(x*2):
        if i == 0:
            i +=1
        for j in range(y*2):
            if j == 0 :
                j+=1
            hillX,hillY,hillSol = HillClimbing(tekrar,step,i,j)
            if bestSolution > hillSol:
                bestSolution = hillSol
                a = hillX
                b = hillY
    return a,b,bestSolution



def HillClimbing(tekrar,step,x,y):
    a = random.uniform(-x,x)
    b = random.uniform(-y,y)
    bestSolution = ackleyFunction(x,y)
    
    for i in range(tekrar):
        newX =  a + random.uniform(-step,step)
        newY = b + random.uniform(-step,step)
        newSolution = ackleyFunction(newX,newY)
        if bestSolution > newSolution:
            a,b = newX,newY
            bestSolution = newSolution

    return a,b,bestSolution

x = int(input("Enter the perimeter length : "))//2
y = int(input("Enter the perimeter width : "))//2    
tekrar = int(input("Please enter the repeat order : "))
step = float(input("Set the step : "))
bestX,bestY,bestSolution = grid(tekrar,step,x,y)
print("best solution is in (X = ",bestX+x," Y = ",bestY+y,") with the value : ",bestSolution)