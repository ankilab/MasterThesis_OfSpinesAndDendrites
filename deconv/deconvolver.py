import os

class Deconvolver:
    def __init__(self, args):
        self.args= args
        self.data_path = os.path.join(os.getcwd(), args['data_path'])
        dir = os.path.join(os.getcwd(), self.data_path, args['result_path'])
        self.res_path = dir

        if not os.path.exists(dir):
            os.makedirs(dir)

    def preprocess(self, **kwargs):
        return NotImplementedError

    def train(self, **kwargs):
        return NotImplementedError

    def predict(self, **kwargs):
        return NotImplementedError

    def predict_img(self, **kwargs):
        return NotImplementedError