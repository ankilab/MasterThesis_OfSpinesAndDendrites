import os
from datetime import datetime


class Deconvolver:
    """
    Super-class for all implemented deconvolution techniques. All derived classes implement a preprocessing, training and
    deconvolution function for either one image or all images within one folder.
    """

    def __init__(self, args):
        """
        Initialize an object providing functionality to deconvolve images.

        :param args: args['source_folder']: Source data location,
                    args['result_path']: file location where results are to be stored,
                    args['data_path']: file location where data is located
        :type args: dict, optional
        """
        self.args= args
        self.args['source_folder'] = args.get('source_folder', './')
        self.data_path = args.get('data_path', './')
        self.data_path=os.path.join(os.getcwd(), self.data_path)
        self.timestamp_created = str(datetime.now()).replace(' ', '_')
        dir =  args.get('result_path', 'Results'+ self.timestamp_created)
        self.res_path = os.path.join(os.getcwd(), dir)

        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)

    def preprocess(self, **kwargs):
        """
        Overwritten in derived classes. Otherwise throws NotImplementedError.
        """

        return NotImplementedError

    def train(self, **kwargs):
        """
        Overwritten in derived classes. Otherwise throws NotImplementedError.
        """

        return NotImplementedError

    def predict(self, **kwargs):
        """
        Overwritten in derived classes. Otherwise throws NotImplementedError.
        """

        return NotImplementedError

    def predict_img(self, **kwargs):
        """
        Overwritten in derived classes. Otherwise throws NotImplementedError.
        """

        return NotImplementedError