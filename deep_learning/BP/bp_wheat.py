from random import random
from math import exp
from csv import reader
from random import randrange


class Database():
    def __init__(self):
        self.dataset = []
        self.dataset_minmax = []

    def load_csv(self, filename):
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                self.dataset.append(row)

    def dataset_str_to_float(self):
        col_len = len(self.dataset[0])
        #print(len(self.dataset))
        for row in self.dataset:
            for col_idx in range(col_len):
                row[col_idx] = float(row[col_idx].strip())

#    def _dataset_minmax(self):
#
#        for col_idx in range(length):
#            value_in_col = [row[col_idx] for row in self.dataset]
#            min_val = min(value_in_col)
#            max_val = max(value_in_col)
#            gap     = max_val - min_val;
#            self.dataset_minmax.append([min_val, max_val, gap])
# use zip() to find min/max value in dataset

    def _dataset_minmax(self):
        self.dataset_minmax = [[min(column), max(column)] for column in zip(*self.dataset)]



    def normalize_dataset(self):
        self._dataset_minmax();
        for col_idx in range(len(self.dataset[0])-1):
            for row in self.dataset: 
                row[col_idx] = (row[col_idx] - self.dataset_minmax[col_idx][0])\
                    / (self.dataset_minmax[col_idx][1] - self.dataset_minmax[col_idx][0])


    def type_to_int(self, column):
        class_values = [row[column] for row in self.dataset]
        unique = set(class_values)

        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i

        for row in self.dataset:
            row[column] = lookup[row[column]]

        

    def get_dataset(self, filename):
        self.load_csv(filename)
        self.dataset_str_to_float()
        self.type_to_int(len(self.dataset[0])-1)
        self.normalize_dataset()
        return self.dataset
        #print(self.dataset)




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


    def cross_validation_split(self, dataset, n_folds):
        dataset_split = []
        dataset_copy = list(dataset)

        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = []
            while len(fold) < fold_size:
                idx = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(idx))
            dataset_split.append(fold)
        return dataset_split
                
                
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0
        
        
    def evaluate_algorithm(self, dataset, learning_rate, epoch, n_folds):
        #self.dataset = dataset
# split data to train set and test set
        folds = self.cross_validation_split(dataset, n_folds)

        scores = []
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])

            test_set = []
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None

            self.train_network(train_set, learning_rate, epoch)
            predicted = self.test_network(test_set)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)

        predicted = self.test_network(test_set)
        actual = [row[-1] for row in fold]
        for i in range(len(predicted)):
            print('predicted:%d, actual: %d' % (predicted[i], actual[i]))
        
        return scores
            
    def test_network(self, test_set):
        predictions = []

        for row in test_set:
            prediction = self.predict(row)
            predictions.append(prediction)

        return predictions

        

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
                #print(each_sample[-1])
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
    print('this program implements wheat detection with the bp algorithm...')

# Construct the dataset for training
    filename = 'seeds_dataset.csv'
    db = Database()
    dataset = db.get_dataset(filename)

# set up the parameters for BP network
    n_input = len(dataset[0]) - 1
    n_hidden = 5
    n_output = len(set(row[-1] for row in dataset))

    bp = Bp_network(n_input, n_hidden, n_output)
    l_rate = 0.3
    n_folds = 5
    epoch = 1000
    bp.init_bp_network()
    scores = bp.evaluate_algorithm(dataset, l_rate, epoch, n_folds)
    print('scores of evaluation algorithm: %s' % scores)

