from blocks.bricks.cost import Cost
from blocks.bricks import application
from theano import tensor


class MAPECost(Cost):

    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        e_y_hat = tensor.abs_(y - y_hat)/y
        return 100*e_y_hat.mean()
