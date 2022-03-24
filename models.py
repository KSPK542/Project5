import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
      
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
       
        return self.w

    def run(self, x):
        
        return nn.DotProduct(x,self.get_weights())

    def get_prediction(self, x):
       
        y = self.run(x)
        if nn.as_scalar(y)<0:
            return -1
        else:
            return 1

    def train(self, dataset):
        
        f=1
        while f==1:
            f=0
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    nn.Parameter.update(self.w,x,nn.as_scalar(y))
                    f=1

class RegressionModel(object):
  
    def __init__(self):
        # Initialize your model parameters here
        "*** CS5368 YOUR CODE HERE ***"
        self.batch_size = 1
        self.w0 = nn.Parameter(1, 50)
        self.b0 = nn.Parameter(1, 50)
        self.w1 = nn.Parameter(50, 1)
        self.b1 = nn.Parameter(1, 1)

    def run(self, x):
        
        xw1 = nn.Linear(x, self.w0)
        r1 = nn.ReLU(nn.AddBias(xw1, self.b0))
        xw2 = nn.Linear(r1, self.w1)
        return nn.AddBias(xw2, self.b1)
    def get_loss(self, x, y):
       
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        
        while True:

            #print(nn.Constant(dataset.x), nn.Constant(dataset.y))

            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                grad = nn.gradients(loss, [self.w0, self.w1, self.b0, self.b1])

                #print(nn.as_scalar(nn.DotProduct(grad[0],grad[0])))
                self.w0.update(grad[0], -0.005)
                self.w1.update(grad[1], -0.005)
                self.b0.update(grad[2], -0.005)
                self.b1.update(grad[3], -0.005)

            #print(nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))))
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02:
                return

class DigitClassificationModel(object):
   
    def __init__(self):
        # Initialize your model parameters here
        "*** CS5368 YOUR CODE HERE ***"
        self.W1 = nn.Parameter(784,250)
        self.b1 = nn.Parameter(1,250)
        self.W2 = nn.Parameter(250,150)
        self.b2 = nn.Parameter(1,150)
        self.W3 = nn.Parameter(150,10)
        self.b3 = nn.Parameter(1,10)

    def run(self, x):
       
        Z1 = nn.AddBias(nn.Linear(x,self.W1),self.b1)
        A1 = nn.ReLU(Z1)
        Z2 = nn.AddBias(nn.Linear(A1,self.W2),self.b2)
        A2 = nn.ReLU(Z2)
        Z3 = nn.AddBias(nn.Linear(A2,self.W3),self.b3)
        return Z3

    def get_loss(self, x, y):
        
        ans = self.run(x)
        return nn.SoftmaxLoss(ans,y)

    def train(self, dataset):
        
        acc = -float('inf')
        while acc<0.976:
            for x,y in dataset.iterate_once(60): 
                grad_wrt_W1, grad_wrt_b1, grad_wrt_W2, grad_wrt_b2, grad_wrt_W3, grad_wrt_b3  = nn.gradients(self.get_loss(x,y), [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
                self.W1.update(grad_wrt_W1, -0.34)
                self.b1.update(grad_wrt_b1, -0.34)
                self.W2.update(grad_wrt_W2, -0.34)
                self.b2.update(grad_wrt_b2, -0.34)
                self.W3.update(grad_wrt_W3, -0.34)
                self.b3.update(grad_wrt_b3, -0.34)
            acc = dataset.get_validation_accuracy()
            print(acc)

class LanguageIDModel(object):
   
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        
        self.W1 = nn.Parameter(self.num_chars,100)
        self.b1 = nn.Parameter(1,100)
        self.W2 = nn.Parameter(100,100)
        self.b2 = nn.Parameter(1,100)
        self.W1_hidden = nn.Parameter(100,100)
        self.b1_hidden = nn.Parameter(1,100)
        self.W2_hidden = nn.Parameter(100,100)
        self.b2_hidden = nn.Parameter(1,100)
        self.W_final = nn.Parameter(100,5)
        self.b_final = nn.Parameter(1,5)

    def run(self, xs):
        
        for i in range(len(xs)):
            if i==0:
                Z1 = nn.AddBias(nn.Linear(xs[i],self.W1),self.b1)
                A1 = nn.ReLU(Z1)
                h = nn.AddBias(nn.Linear(A1,self.W2),self.b2)
            else:
                Z_one = nn.AddBias(nn.Add(nn.Linear(xs[i], self.W1), nn.Linear(h, self.W1_hidden)),self.b1_hidden)
                A_one = nn.ReLU(Z_one)
                Z_two = nn.AddBias(nn.Linear(A_one,self.W2_hidden),self.b2_hidden)
                h = nn.ReLU(Z_two)
        return nn.AddBias(nn.Linear(h,self.W_final),self.b_final)

    def get_loss(self, xs, y):
        
        ans = self.run(xs)
        return nn.SoftmaxLoss(ans,y)


    def train(self, dataset):
        
        acc = -float('inf')
        for i in range(21):
            for x,y in dataset.iterate_once(60): 
                grad_wrt_W1, grad_wrt_b1, grad_wrt_W2, grad_wrt_b2, grad_wrt_W1_hidden, grad_wrt_b1_hidden, grad_wrt_W2_hidden, grad_wrt_b2_hidden, grad_wrt_W_final, grad_wrt_b_final = nn.gradients(self.get_loss(x,y), [self.W1, self.b1, self.W2, self.b2, self.W1_hidden, self.b1_hidden, self.W2_hidden, self.b2_hidden, self.W_final, self.b_final])
                self.W1.update(grad_wrt_W1, -0.15)
                self.b1.update(grad_wrt_b1, -0.15)
                self.W2.update(grad_wrt_W2, -0.15)
                self.b2.update(grad_wrt_b2, -0.15)
                self.W1_hidden.update(grad_wrt_W1_hidden, -0.15)
                self.b1_hidden.update(grad_wrt_b1_hidden, -0.15)
                self.W2_hidden.update(grad_wrt_W2_hidden, -0.15)
                self.b2_hidden.update(grad_wrt_b2_hidden, -0.15)
                self.W_final.update(grad_wrt_W_final, -0.15)
                self.b_final.update(grad_wrt_b_final, -0.15)
            acc = dataset.get_validation_accuracy()
