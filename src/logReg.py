import math

# function for one interation of gradient descent

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def gradientDescent(data, w1, w2, b, m):
    J = 0
    dw1 = 0
    dw2 = 0
    db = 0
    for i in range(m):
        zi = w1 * data[i][1] + w2 * data[i][2] + b
        ai = sigmoid(zi)
        if ai == 1.0:
            ai = 0.999999999
        elif ai == -1.0:
            ai = -0.999999999       
        J += (-(data[i][0] * math.log10(ai) + (1-data[i][0]) * (math.log10(1 - ai))))
    
        dzi = ai - data[i][0]
        dw1 += data[i][1] * dzi
        dw2 += data[i][2] * dzi
        db += dzi

    J /= m
    dw1 /= m
    dw2 /= m
    db /= m

    return [J, dw1, dw2, db]

# read dataset for important variables and data, init other important variables

dataFile = open('src\dataset.txt', 'r')
dataArr = dataFile.read().split('\n')
m = int(len(dataArr))

for i in range(len(dataArr)):
    dataArr[i] = dataArr[i].split(',')
    if dataArr[i][0] == "\"Male\"":
        dataArr[i][0] = 1
    else: 
        dataArr[i][0] = 0
    dataArr[i][1] = float(dataArr[i][1])
    dataArr[i][2] = float(dataArr[i][2])

# run given some interation count

w1 = 0
w2 = 0
b = 0

iterationCount = 2000
learningRate = 0.04

log = open('log.txt', 'w')

for i in range(iterationCount):
    retArray = gradientDescent(dataArr, w1, w2, b, m)
    w1 = w1 - learningRate * retArray[1]
    w2 = w2 - learningRate * retArray[2]
    b = b - learningRate * retArray[3]

    log.write("(" + str(i / 100) + ", " + str(round(w1, 3)) + ")\n")
    log.write("(" + str(i / 100) + ", " + str(round(w2, 3)) + ")\n")
    log.write("(" + str(i / 100) + ", " + str(round(b, 3)) + ")\n")
    #log.write(str(round(w1, 3)) + ' ' + str(round(w2, 3)) + ' ' + str(round(b, 3)) + '\n')