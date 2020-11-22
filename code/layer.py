# 5.4 단순한 계층 구성하기
# 사과 쇼핑

class Layer:
    
    def __init__(self):
        self.X = None
        self.Y = None

    def mul(self, X, Y):
        """
        docstring
        """
        self.X = X
        self.Y = Y

        return self.X, * self.Y

    def add(self, X, Y):
        """
        docstring
        """
        self.X = X
        self.Y = Y

        return self.X, + self.Y

    def backward(self, dout):

        self.dX = dout*self.X
        self.dY = dout*self.Y

        return self.dX, self,dY

apple = 100
orange = 150

apple_num = 2
orange_num = 3

tax = 1.1

mul_apple_layer = Layer()