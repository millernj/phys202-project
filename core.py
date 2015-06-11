import numpy as np

sigmoid = lambda x: 1/(1 +np.exp(-x))

def perceptron_sigmoid(weights, inputvect):
    return sigmoid(np.dot(np.append(inputvect,[1]), weights))

def gen_network(size):
    weights= [np.array([[np.random.randn() for _ in range(size[n-1]+1)]
               for _ in range(size[n])]) for n in range(len(size))[1:]]
    return weights

def propforward(network, inputvect):
    outputs = []
    for layer in network:
        neural_input = inputvect
        output = [perceptron_sigmoid(weights, neural_input) for weights in layer]
        outputs.append(output)
        inputvect = output
    
    outputs = np.array(outputs)
    return [outputs[:-1], outputs[-1]]

def target_convert(n):
    result = np.zeros((10,))
    result[n]=1
    return result

def find_deltas_sigmoid(outputs, targets):
    return [output*(1-output)*(output-target) for output, target in zip(outputs, targets)]

def edit_weights(layer, input_list, deltas, learning_rate):          
        for a, inpt in enumerate(input_list):
            layer-=learning_rate/len(input_list)*np.dot(deltas[a].reshape(len(deltas[a]),1),
                                                        np.append(inpt,[1]).reshape(1,len(inpt)+1))
def backprob(network, inputvect, targets):
    
    hidden_outputs, outputs = propforward(network, inputvect)
    
    change_in_outputs = find_deltas_sigmoid(outputs, targets)
    
    list_deltas = [[] for _ in range(len(network))]
    list_deltas[-1] = change_in_outputs
    
    for n in range(len(network))[-1:0:-1]:
        delta = change_in_outputs
        change_in_hidden_outputs= [hidden_output*(1-hidden_output)*
                               np.dot(delta, np.array([a[i] for a in network[n]]).transpose())
                               for i, hidden_output in enumerate(hidden_outputs[n-1])]
        list_deltas[n-1] = change_in_hidden_outputs
        change_in_outputs = change_in_hidden_outputs
    
    return list_deltas

def stoc_descent(network, input_list, target_list, learning_rate):
    mega_delta = []
    hidden_output = [propforward(network, inpt)[0] for inpt in input_list]
    for inpt, target in zip(input_list, target_list):
        mega_delta.append(backprob(network, inpt, target))
    
    inputs=[]
    inputs.append(input_list)
    for n in range(len(network)):
        inputs.append(hidden_output[n])
    assert len(inputs) == len(network) + 1
    deltas = []
    
    

    for n in range(len(network)):
        deltas.append([np.array(delta[n]) for delta in mega_delta])
        
    assert len(deltas)==len(network)
    for n in range(len(network)):
        edit_weights(network[n], inputs[n], deltas[n], learning_rate)

def output_reader(output):
    assert len(output)==10
    result=[]
    for i, t in enumerate(output):
        if t == max(output) and abs(t-1)<=0.5:
            result.append(i)
    if len(result)==1:
        return result[0]
    else:
        return 0

def target_convert(n):
    assert n <= 9 and n >= 0
    n = round(n)
    result = np.zeros((10,))
    result[n]=1
    return result

def train_network(network, training_inputs, training_targets, training_cycles = 30,
                  numbers_per_cycle = 1438,batch_size = 15,learning_rate = 1):
    
    train_data_index = np.linspace(0,numbers_per_cycle, numbers_per_cycle + 1)
    target_list = [target_convert(n) for n in training_targets[0:numbers_per_cycle]]
    np.random.seed(1)
    np.random.shuffle(train_data_index)
    for _ in range(training_cycles):
        for n in train_data_index:
            if n+batch_size <= numbers_per_cycle:
                training_data = training_inputs[int(n):int(n+batch_size)]
                target_data = target_list[int(n):int(n+batch_size)]
            else: 
                training_data = training_inputs[int(n-batch_size):numbers_per_cycle]
                assert len(training_data)!=0
                target_data = target_list[int(n-batch_size):numbers_per_cycle]
            stoc_descent(network, training_data, target_data, learning_rate)
                  
def check_net(network, testing_list, target_list, rnge):
    guesses = []
    targets = []
    number_correct = 0
    rnge = range(rnge[0],rnge[1])
    for n in rnge:

        guesses.append(output_reader(propforward(network, testing_list[n])[1]))
        targets.append(target_list[n])

    for guess, target in zip(guesses, targets):
        if guess == target:
            number_correct+=1
    number_total = len(rnge)
    print(number_correct/number_total*100)
    print("%s/%s" %(str(number_correct), str(number_total)))

