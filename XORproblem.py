x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
test = np.array([[1,0],[0,1],[0,0]])
epoch = 10000
rate = 0.1
iNeu = 2 
hNeu = 2
oNeu = 1
weightHidden = np.random.uniform(size=(iNeu,hNeu))
weightOut = np.random.uniform(size=(hNeu,oNeu))
biasHidden = np.random.uniform(size=(1,hNeu))
biasOut = np.random.uniform(size=(1,oNeu))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_Moshtaq(x):
    return sigmoid(x) * (1 - x)
for _ in range(epoch):
    
    hidden = sigmoid(np.dot(x, weightHidden) + biasHidden) 
    output = sigmoid(np.dot(hidden, weightOut) + biasOut)
    
    errorOut = y - output
    d_prediction = errorOut * sigmoid_Moshtaq(output)
    errorHidden = d_prediction.dot(wOut.T)
    d_hidden = errorHidden * sigmoid_Moshtaq(hidden)
    
    weightOut += hidden.T.dot(d_prediction) * rate
    weightHidden += x.T.dot(d_hidden) * rate
    biasHidden += np.sum(d_hidden, axis=0, keepdims=True) * rate
    biasOut += np.sum(d_prediction, axis=0, keepdims=True) * rate
    
print(f"Final Weight Hiddens: {weightHidden} , Bias Hiddens: {biasHidden} ,Final Weight Out:{weightOut},Bias Out:{biasOut}")

def Per(test, weightOut, weightHidden, biasOut, biasHidden):
    hidden = sigmoid(np.dot(test, weightHidden) + biasHidden)
    output = sigmoid(np.dot(hidden, weightOut) + biasOut)
    per = [0] * len(test)
    c = 0 
    for i in output:
        if np.round(i) > 0 :
            per[c] = 1
        else :
            per[c] = 0
        c +=1 
    return per        
f = [(x,y) for x,y in zip(test,Per(test, weightOut, weightHidden, biasOut, biasHidden))]
print(f)