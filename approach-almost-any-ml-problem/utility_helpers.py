####################################################DESCRIPTION############################################
#
import matplotlib.pyplot as plt
## All these classes are used as utility classes

#
####################################################DESCRIPTION############################################


# import required packages
from sklearn import datasets
from sklearn.model_selection import KFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn


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


class Metrics:
    def __init__(self, threshold=0.5):
        self.log_loss_score = None
        self.roc_score = None
        self.roc_auc_fpr_list = None
        self.roc_auc_tpr_list = None
        self.false_positive_rate = None
        self.f1_score = None
        self.plot_precision_recall_curve = None
        self.empty_recall_plot_list = None
        self.empty_precession_plot_list = None
        self.true_positive_rate = None
        self.recall = None
        self.precession = None
        self.accuracy_score = None
        # self.true = true
        # self.predictions = predictions
        self.threshold = threshold
        # self.threshold_list = threshold_list
        self.accuracy_counter = 0
        # self.true_positive_counter = 0
        # self.true_negative_counter = 0
        # self.false_positive_counter = 0
        # self.false_negative_counter = 0

    def compute_accuracy(self, true, predictions):
        try:
            for yt, yp in zip(true, predictions):
                if yt == yp:
                    self.accuracy_counter += 1

            return np.round((self.accuracy_counter / len(true)) * 100, 5)

        except Exception as e:
            raise e

    def compute_true_positive(self, true, predictions):
        try:
            true_positive_counter = 0
            for yt, yp in zip(true, predictions):
                if yt == 1 and yp == 1:
                    true_positive_counter += 1

            return true_positive_counter

        except Exception as e:
            raise e

    def compute_true_negative(self, true, predictions):
        try:
            true_negative_counter = 0
            for yt, yp in zip(true, predictions):
                if yt == 0 and yp == 0:
                    true_negative_counter += 1

            return true_negative_counter

        except Exception as e:
            raise e

    def compute_false_positive(self, true, predictions):
        try:
            false_positive_counter = 0
            for yt, yp in zip(true, predictions):
                if yt == 0 and yp == 1:
                    false_positive_counter += 1

            return false_positive_counter

        except Exception as e:
            raise e

    def compute_false_negative(self, true, predictions):
        try:
            false_negative_counter = 0
            for yt, yp in zip(true, predictions):
                if yt == 1 and yp == 0:
                    false_negative_counter += 1

            return false_negative_counter

        except Exception as e:
            raise e

    def compute_accuracy_score(self, true, predictions):
        try:
            tp_ = self.compute_true_positive(true, predictions)
            fp_ = self.compute_false_positive(true, predictions)
            fn_ = self.compute_false_negative(true, predictions)
            tn_ = self.compute_true_negative(true, predictions)

            self.accuracy_score = (tp_ + tn_) / (tp_ + tn_ + fn_ + fp_)

            return np.round(self.accuracy_score * 100, 5)

        except Exception as e:
            raise e

    def compute_precision(self, true, predictions):
        try:
            tp_ = self.compute_true_positive(true, predictions)
            fp_ = self.compute_false_positive(true, predictions)
            self.precession = tp_ / (tp_ + fp_)

            return self.precession

        except Exception as e:
            raise e

    def compute_recall(self, true, predictions):
        try:
            tp_ = self.compute_true_positive(true, predictions)
            fn_ = self.compute_false_negative(true, predictions)
            self.recall = tp_ / (tp_ + fn_)

            return self.recall

        except Exception as e:
            raise e

    def compute_precision_recall_curve(self, true, predictions, threshold_list):
        try:
            self.empty_precession_plot_list = []
            self.empty_recall_plot_list = []

            for threshold in threshold_list:
                temp_prediction = [1 if prob_ >= threshold else 0 for prob_ in predictions]
                pre_ = self.compute_precision(true, temp_prediction)
                rec_ = self.compute_recall(true, temp_prediction)

                self.empty_precession_plot_list.append(pre_)
                self.empty_recall_plot_list.append(rec_)

            print(self.empty_recall_plot_list)
            print(self.empty_precession_plot_list)
            plt.figure(figsize=(10, 10))
            plt.grid()
            plt.plot(self.empty_recall_plot_list, self.empty_precession_plot_list, color='orange', lw=2, label='P-R Curve')
            plt.xlabel('Recall', fontsize=20)
            plt.ylabel('Precision', fontsize=20)
            plt.title("Precision Recall Curve", fontsize=20)
            plt.legend(fontsize='large')
            plt.show()

        except Exception as e:
            raise e

    def compute_f1_score(self, true, predictions):
        try:
            re_ = self.compute_recall(true, predictions)
            pre_ = self.compute_precision(true, predictions)

            self.f1_score = 2 * pre_ * re_ / (pre_ + re_)

            return self.f1_score

        except Exception as e:
            raise e

    def compute_true_positive_rate(self, true, predictions):
        try:
            tp_ = self.compute_true_positive(true, predictions)
            fn_ = self.compute_false_negative(true, predictions)
            self.true_positive_rate = tp_ / (tp_ + fn_)

            return self.true_positive_rate

        except Exception as e:
            raise e

    def compute_false_positive_rate(self, true, predictions):
        try:
            fp_ = self.compute_false_positive(true, predictions)
            tn_ = self.compute_true_negative(true, predictions)

            self.false_positive_rate = fp_ / (fp_ + tn_)

            return self.false_positive_rate

        except Exception as e:
            raise e

    def compute_roc_auc_curve(self, true, predictions, thresholds):
        try:
            self.roc_auc_tpr_list = []
            self.roc_auc_fpr_list = []

            for threshold in thresholds:
                temp_preds_ = [1 if prob_ >= threshold else 0 for prob_ in predictions]
                tpr_ = self.compute_true_positive_rate(true, temp_preds_)
                fpr_ = self.compute_false_positive_rate(true, temp_preds_)

                self.roc_auc_fpr_list.append(fpr_)
                self.roc_auc_tpr_list.append(tpr_)

            plt.figure(figsize=(10,10))
            plt.fill_between(self.roc_auc_fpr_list, self.roc_auc_tpr_list, alpha=0.4, color='lightpink')
            plt.plot(self.roc_auc_fpr_list, self.roc_auc_tpr_list, color='black', lw=2, label="Our Model")
            plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Model')
            plt.grid()
            plt.title('ROC Curve', fontsize=20)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel("FPR", fontsize=20)
            plt.ylabel("TPR", fontsize=20)
            plt.legend(fontsize='large')
            plt.show()

        except Exception as e:
            raise e

    def compute_roc_auc_score(self, true, predictions):
        try:
            self.roc_score = sklearn.metrics.roc_auc_score(true, predictions)

            return self.roc_score

        except Exception as e:
            raise e

    def compute_log_loss(self, true, proba):
        try:
            self.log_loss_score = sklearn.metrics.log_loss(true, proba)

            return self.log_loss_score

        except Exception as e:
            raise e

