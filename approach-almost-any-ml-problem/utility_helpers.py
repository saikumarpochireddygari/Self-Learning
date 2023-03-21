####################################################DESCRIPTION############################################
#

## All these classes are used as utility classes

#
####################################################DESCRIPTION############################################


# import required packages
from sklearn import datasets


class LoadDataSets:
    def __init__(self):
        self.wine_data = None
        self.iris_data = None
        self.mnist_data = None

    def get_mnist_data(self):
        """
        This function loads the mnist data into the kernel
        :returns/op:
        It returns data, labels
        """
        try:
            self.mnist_data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
        except Exception as e:
            return e

        return self.mnist_data

    def get_iris_data(self):
        """
        This function loads the iris data into the kernel
        :returns/op:
        It returns data, labels
        """
        try:
            self.iris_data = datasets.load_iris(return_X_y=True)
        except Exception as e:
            return e

        return self.iris_data

    def get_wine_data(self):
        """
        This function loads the wine data into the kernel
        :return:
        It returns data, labels
        """
        try:
            self.wine_data = datasets.load_wine(return_X_y=True)
        except Exception as e:
            return e

        return self.wine_data


#pick up from here
class KFoldCrossValidation:
    pass