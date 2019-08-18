from random import random
from math import exp

class Bp_network():
    
    def __init__(self, input_num, hidden_num, output_num):
        self.input_num  = input_num;
        self.hidden_num = hidden_num;
        self.output_num = output_num;
        self.hidden_layer = []
        self.output_layer = []

    # construct the bp network with initial random weights values.    
    def init_bp_network(self):
        
        for i in range(self.hidden_num):
            #temp = {'weight':[ 1.0 for i in range(self.input_num + 1)]}
            temp = {'weight':[random() for i in range(self.input_num + 1)]}
            self.hidden_layer.append(temp)

        for i in range(self.output_num):
            #temp = {'weight':[1.0 for i in range(self.hidden_num + 1)]}
            temp = {'weight':[random() for i in range(self.hidden_num + 1)]}            
            self.output_layer.append(temp)
            

    def acti_func(self, acti_value):
        return 1.0 / (1.0 + exp(-acti_value))

        
    def forward_propagate(self, single_sample):
       # print(single_sample)

        # calculate the output value in hidden layer
        for n_hidden_point in range(self.hidden_num):
            activation = self.hidden_layer[n_hidden_point]['weight'][-1]

            for i in range(self.input_num):
                activation += single_sample[i] * self.hidden_layer[n_hidden_point]['weight'][i]

            self.hidden_layer[n_hidden_point]['output'] = self.acti_func(activation)

        # calculate the output value in output layer
        for n_output_point in range(self.output_num):
            activation = self.output_layer[n_output_point]['weight'][-1]

            for i in range(self.hidden_num):
               # print(self.hidden_layer[i]['output'])
                activation += self.hidden_layer[i]['output'] * self.output_layer[n_output_point]['weight'][i]

            #print(activation)
            self.output_layer[n_output_point]['output'] = self.acti_func(activation)

        return [ self.output_layer[n]['output'] for n in range(self.output_num) ]
      

    def back_propagate(self, ref_output_value):
        # calculate the responsibility value in output layer
        for n_output_point in range(self.output_num):
            out_value = self.output_layer[n_output_point]['output']
            error = ref_output_value[n_output_point] - out_value
            self.output_layer[n_output_point]['responsibility'] = error * out_value * (1 - out_value)

            
         # calculate the responsibility value in hidden layer
        for n_hidden_point in range(self.hidden_num):
            self.hidden_layer[n_hidden_point]['responsibility'] = 0.0
            out_value = self.hidden_layer[n_hidden_point]['output']

            for n_output_point in range(self.output_num):
                temp_responsibility = self.output_layer[n_output_point]['responsibility']
                temp_weight = self.output_layer[n_output_point]['weight'][n_hidden_point]
                #print(temp_responsibility)
                #print(temp_weight)
                self.hidden_layer[n_hidden_point]['responsibility'] += temp_responsibility * temp_weight

            #print(self.hidden_layer[n_hidden_point]['responsibility'])
            self.hidden_layer[n_hidden_point]['responsibility'] *= out_value * (1 - out_value)

   
        
    def update_weights(self, each_sample, rate):

        # update the weights of output points
        for n_output_point in range(self.output_num):
            #temp_weight = self.output_layer[n_output_point]['weight']
            temp_responsibility = self.output_layer[n_output_point]['responsibility']
            
            for n_hidden_point in range(self.hidden_num):
                temp_output = self.hidden_layer[n_hidden_point]['output']
                self.output_layer[n_output_point]['weight'][n_hidden_point] += rate * temp_responsibility * temp_output

            self.output_layer[n_output_point]['weight'][-1] += rate * temp_responsibility    
        
         # update the weights of hidden points
        for n_hidden_point in range(self.hidden_num):
            #temp_weight = self.output_layer[n_output_point]['weight']
            temp_responsibility = self.hidden_layer[n_hidden_point]['responsibility']
            
            for n_input_point in range(self.input_num):
                self.hidden_layer[n_hidden_point]['weight'][n_input_point] += rate * temp_responsibility * each_sample[n_input_point]

            self.hidden_layer[n_hidden_point]['weight'][-1] += rate * temp_responsibility    

      #  print('hidden_layer:')
      #  print(self.hidden_layer)
      #  print('output layer:')
      #  print(self.output_layer)


        
    def train_network(self, dataset, learning_rate, epoch):
        for e in range(epoch):
            sum_error = 0.0
            for each_sample in dataset:
                #print(each_sample)
                outputs = self.forward_propagate(each_sample)
                
                # Basic understanding here:
                # n output points in bp network represent the probability of n outputs.
                # if n = 5 and the result in the dataset is 2, then:
                # output point 0 with probability: 0%
                # output point 1 with probability: 0%
                # output point 2 with probability: 100%
                # output point 3 with probability: 0%
                # output point 4 with probability: 0%
                
                expected = [0 for i in range(self.output_num)]
                expected[each_sample[-1]] = 1

                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                
                self.back_propagate(expected)
                self.update_weights(each_sample, learning_rate)
            #print('period: %d, error: %.3f ' % (e, sum_error))

    def predict(self, single_sample):
        self.forward_propagate(single_sample)
        temp = [ self.output_layer[n]['output'] for n in range(self.output_num) ]


        #print(temp)
        return temp.index(max(temp))

           


if __name__ == '__main__':
    print('this program implements the bp algorithm...')
    dataset = [[1, 1, 0],
               [1, 0, 1],
               [0, 1, 1],
               [0, 0, 0]
        ]
    #dataset = [[1, 1, 0]]

    n_input  = len(dataset[0]) - 1
    n_output = len(set(row[-1] for row in dataset))
    n_hidden = 2
    l_rate = 0.1
    epoch= 6000

    bp = Bp_network(n_input, n_hidden, n_output)
    bp.init_bp_network()
    bp.train_network(dataset, l_rate, epoch)

    for layer in dataset:
        result = bp.predict(layer)
        print('predicted = %d, actual value = %d' % (result, layer[-1]))
    
