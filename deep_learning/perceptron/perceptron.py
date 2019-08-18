class Perceptron():
    def __init__(self, input_para_num, acti_func):
        self.activator = acti_func
        self.weights = [0.0 for _ in range(input_para_num)]

    def train(self, dataset, iteration, rate):
        for i in range(iteration):
            for data_row in dataset:
                prediction = self.predict(data_row)
                self.update_weights(prediction, data_row, rate)

    def predict(self, one_data):
        lenth = len(self.weights)
        predict = 0.0
        
        for i in range(lenth):
            predict += self.weights[i] * one_data[i]
            
        return self.activator(predict);

    def update_weights(self, predicted_value, one_data, rate):
        delta = rate * (one_data[-1] - predicted_value) 

        for i in range(len(self.weights)):
            self.weights[i] += delta * one_data[i]

    def print_weight(self):
        print ('w0: %04f, w1: %04f, w2: %04f' % (self.weights[0], self.weights[1], self.weights[2]))
        
def func_activator(input_value):
    return 1.0 if input_value >= 0.0 else 0.0


if __name__ == '__main__':
    print('This program implements the perceptron...')

    dataset = [[-1, 0, 0, 0], [-1, 0, 1, 0], [-1, 1, 0, 0], [-1, 1, 1, 1]]
    p = Perceptron(3, func_activator)
    
    p.train(dataset, 1000, 0.1)

    p.print_weight()
    
    a = [-1, 1, 1]
    b = [-1, 1, 0]
    c = [-1, 0, 1]
    d = [-1, 0, 0]
    
    a_ = p.predict(a)
    b_ = p.predict(b)
    c_ = p.predict(c)
    d_ = p.predict(d)

    print(a_)
    print(b_)
    print(c_)
    print(d_)
