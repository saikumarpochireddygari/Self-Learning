####################################################DESCRIPTION############################################
#

## All these classes are used as utility classes

#
####################################################DESCRIPTION############################################


# import required packages
from sklearn import datasets
from sklearn.model_selection import KFold


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
            raise e

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
            raise e

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
            raise e

        return self.wine_data


class KFoldCrossValidation:
    def __init__(self, data_frame):
        self.kf = None
        self.df_ = None
        self.target = None
        self.data_frame = data_frame

    def return_k_fold_dataset(self, no_of_splits):
        try:
            self.df_ = self.data_frame
            self.df_['kfold'] = -1

            self.df_ = self.df_.sample(frac=1).reset_index(drop=True)
            self.kf = KFold(n_splits=no_of_splits)

            for fold, (trn_, val_) in enumerate(self.kf.split(X=self.df_)):
                self.df_.loc[val_, 'kfold'] = fold

            return self.df_

        except Exception as e:
            raise e

    def return_stratified_kfold_dataset(self, no_of_splits, target_feature):
        try:
            self.df_ = self.data_frame
            self.df_['kfold'] = -1
            self.target = self.df_[target_feature]

            self.df_ = self.df_.sample(frac=1).reset_index(drop=True)
            self.kf = KFold(n_splits=no_of_splits)

            for fold, (trn_, val_) in enumerate(self.kf.split(X=self.df_, y=self.target)):
                self.df_.loc[val_, 'kfold'] = fold

            return self.df_

        except Exception as e:
            raise e

