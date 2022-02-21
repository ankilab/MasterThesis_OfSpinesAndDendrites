import os
from datetime import datetime

class Deconvolver:
    def __init__(self, args):
        self.args= args
        self.data_path = args.get('data_path', './')
        self.data_path=os.path.join(os.getcwd(), self.data_path)
        self.timestamp_created = str(datetime.now()).replace(' ', '_')
        dir =  args.get('result_path', 'Results'+ self.timestamp_created)
        self.res_path = os.path.join(os.getcwd(), dir)

        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)

    def preprocess(self, **kwargs):
        return NotImplementedError

    def train(self, **kwargs):
        return NotImplementedError

    def predict(self, **kwargs):
        return NotImplementedError

    def predict_img(self, **kwargs):
        return NotImplementedError