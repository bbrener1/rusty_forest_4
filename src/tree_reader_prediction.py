import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

class Prediction:

    def __init__(self, forest, matrix):
        self.forest = forest
        self.matrix = matrix
        self.mode = None
        self.nse = None
        self.nme = None
        self.nae = None
        self.nwp = None
        self.smc = None
        self.factors = None

    def node_sample_encoding(self):
        if self.nse is None:
            self.nse = self.forest.predict_node_sample_encoding(
                self.matrix, leaves=False)
        return self.nse

    def node_mean_encoding(self):
        if self.nme is None:
            self.nme = self.forest.mean_matrix(self.forest.nodes())
        return self.nme

    def node_additive_encoding(self):
        if self.nae is None:
            self.nae = self.forest.mean_additive_matrix(self.forest.nodes())
        return self.nae

    def additive_prediction(self, depth=8):
        encoding = self.node_sample_encoding().T
        feature_predictions = self.node_additive_encoding().T
        prediction = np.dot(encoding.astype(dtype=int), feature_predictions)
        prediction /= len(self.forest.trees)

        return prediction

    def mean_prediction(self):
        leaf_mask = self.forest.leaf_mask()
        encoding_prediction = self.node_sample_encoding()[leaf_mask].T
        feature_predictions = self.node_mean_encoding()[leaf_mask]
        scaling = np.dot(encoding_prediction,
                         np.ones(feature_predictions.shape))

        prediction = np.dot(encoding_prediction, feature_predictions) / scaling
        prediction[scaling == 0] = 0
        return prediction

    def prediction(self, mode=None):

        if mode is None:
            mode = self.mode
        if mode is None:
            mode = "additive_mean"
        self.mode = mode

        if mode == "additive_mean":
            prediction = self.additive_prediction()
        elif mode == "mean":
            prediction = self.mean_prediction()
        else:
            raise Exception(f"Not a valid mode {mode}")

        return prediction

    def residuals(self, truth=None, mode=None):

        if truth is None:
            truth = self.matrix

        prediction = self.prediction(mode=mode)
        residuals = truth - prediction

        return residuals

    def null_residuals(self, truth=None):

        if truth is None:
            truth = self.matrix

        centered_truth = truth - np.mean(truth, axis=0)

        return centered_truth

    def node_residuals(self, node, truth=None):

        if truth is None:
            truth = self.matrix

        sample_predictions = self.node_sample_encoding()[node.index]
        feature_predictions = self.node_mean_encoding()[node.index]
        residuals = truth[sample_predictions] - feature_predictions

        return residuals

    def node_feature_remaining_error(self, nodes):

        per_node_fraction = []

        for node in nodes:

            if node.parent is not None:

                node_residuals = self.node_residuals(node)
                remaining_error = np.sum(np.power(node_residuals, 2), axis=0)

                sister_residuals = self.node_residuals(node.sister())
                remaining_error += np.sum(np.power(sister_residuals, 2), axis=0)

                parent_residuals = self.node_residuals(node.parent)
                original_error += np.sum(np.power(parent_residuals, 2), axis=0)

                # Avoid nans:
                # (there's gotta be a better way) *billy mays theme starts*

                remaining_error += 1
                original_error += 1

                per_node_fraction.append(remaining_error / original_error)

            else:
                per_node_fraction.append(1)

        return np.mean(np.array(per_node_fraction), axis=0)

    def sample_clusters(self):

        if self.smc is None:

            leaf_mask = self.forest.leaf_mask()
            encoding_prediction = self.node_sample_encoding()[leaf_mask].T
            leaf_means = np.array([l.sample_cluster_means() for l in self.forest.leaves()])
            scaling = np.dot(encoding_prediction,
                             np.ones(leaf_means.shape))

            prediction = np.dot(encoding_prediction, leaf_means) / scaling
            prediction[scaling == 0] = 0

            self.smc = np.argmax(prediction, axis=1)

        return self.smc

    def factor_matrix(self):
        predicted_encoding = self.node_sample_encoding()
        predicted_factors = np.zeros(
            (self.matrix.shape[0], len(self.forest.split_clusters)))
        predicted_factors[:, 0] = 1.
        for i in range(1, len(self.forest.split_clusters[0:])):
            predicted_factors[:, i] = self.forest.split_clusters[i].predict_sister_scores(
                predicted_encoding)
        return predicted_factors

    def compare_sample_clusters(self, other):

        self_samples = self.sample_clusters()
        other_samples = other.sample_clusters()

        plt.figure()
        plt.title("Sample Cluster Frequency, Self vs Other")
        plt.xlabel("Cluster")
        plt.ylabel("Frequency")
        plt.xticks(np.arange(len(self.forest.sample_clusters)))
        plt.hist(self_samples, alpha=.5, density=True, label="Self",
                 bins=np.arange(len(self.forest.sample_clusters) + 1))
        plt.hist(other_samples, alpha=.5, density=True, label="Other",
                 bins=np.arange(len(self.forest.sample_clusters) + 1))
        plt.legend()
        plt.show()
        pass

    def compare_factors(self, other, no_plot=False, log=True, bins=20):

        from scipy.stats import entropy
        from scipy.stats import ks_2samp
        from scipy.stats import ranksums

        own_factors = self.factor_matrix()
        other_factors = other.factor_matrix()

        symmetric_entropy = []
        bin_interval = 2. / bins

        for i, (own_f, other_f) in enumerate(zip(own_factors.T, other_factors.T)):
            own_hist = np.histogram(
                own_f, bins=np.arange(-1, 1, bin_interval))[0] + 1
            other_hist = np.histogram(
                other_f, bins=np.arange(-1, 1, bin_interval))[0] + 1
            own_prob = own_hist / np.sum(own_hist)
            other_prob = other_hist / np.sum(other_hist)
        #     print("##############################")
            forward_entropy = entropy(own_prob, qk=other_prob)
            reverse_entropy = entropy(other_prob, qk=own_prob)
            average_entropy = (forward_entropy + reverse_entropy) / 2
            symmetric_entropy.append(average_entropy)
            print(f"{i} Entropy: {average_entropy}")
            ks = ks_2samp(own_f, other_f)
            print(f"Kolmogorov-Smirnov: {ks}")
            mwu = ranksums(own_f, other_f)
            print(f"Rank Sum: {mwu}")
            if not no_plot:
                own_log_prob = np.log(own_hist / np.sum(own_hist))
                other_log_prob = np.log(other_hist / np.sum(other_hist))

                lin_min = np.min(
                    [np.min(own_log_prob), np.min(other_log_prob)])

                plt.figure(figsize=(5, 5))
                plt.title(f"{self.forest.split_clusters[i].name()} Comparison")
                plt.scatter(own_log_prob, other_log_prob,
                            c=np.arange(-1, 1, bin_interval)[:-1], cmap='seismic')
                plt.plot([0, lin_min], [0, lin_min], color='red', alpha=.5)
                plt.xlabel("Factor Frequency, Self (Log Probability)")
                plt.ylabel("Factor Frequency, Other (Log Probability)")
                plt.colorbar(label="Factor Value")
                plt.show()

        return symmetric_entropy

    def prediction_report(self, truth=None, n=10, mode="additive_mean", no_plot=False):

        null_square_residuals = np.power(self.null_residuals(truth=truth), 2)
        null_residual_sum = np.sum(null_square_residuals)

        forest_square_residuals = np.power(self.residuals(truth=truth), 2)
        predicted_residual_sum = np.sum(forest_square_residuals)

        explained = predicted_residual_sum / null_residual_sum

        print(explained)

        # Add one here to avoid divisions by zero, but this is bad
        # Need better solution

        null_feature_residuals = np.sum(null_square_residuals, axis=0) + 1
        forest_feature_residuals = np.sum(forest_square_residuals, axis=0) + 1

        features_explained = forest_feature_residuals / null_feature_residuals

        if not no_plot:
            plt.figure()
            plt.title("Distribution of Target Coefficients of Determination")
            plt.hist(features_explained, bins=np.arange(0, 1, .05), log=True)
            plt.xlabel("CoD")
            plt.ylabel("Frequency")
            plt.show()

        feature_sort = np.argsort(features_explained)

        print(
            (self.forest.output_features[feature_sort[:n]], features_explained[feature_sort[:n]]))
        print((self.forest.output_features[feature_sort[-n:]],
               features_explained[feature_sort[-n:]]))

        null_sample_residuals = np.sum(null_square_residuals, axis=1) + 1
        forest_sample_residuals = np.sum(forest_square_residuals, axis=1) + 1

        samples_explained = forest_sample_residuals / null_sample_residuals

        sample_sort = np.argsort(samples_explained)

        print(sample_sort[:n], samples_explained[sample_sort[:n]])
        print(sample_sort[-n:], samples_explained[sample_sort[-n:]])

        if not no_plot:
            plt.figure()
            plt.title("Distribution of Sample Coefficients of Determination")
            plt.hist(samples_explained, bins=np.arange(0, 1, .05), log=True)
            plt.xlabel("CoD")
            plt.ylabel("Frequency")
            plt.show()

        return features_explained, samples_explained

    def feature_mse(self, truth=None, mode='additive_mean'):

        residuals = self.residuals(truth=truth, mode=mode)
        mse = np.mean(np.power(residuals, 2), axis=0)

        return mse

    def jackknife_feature_mse_variance(self, mode='additive_mean'):

        squared_residuals = np.power(self.residuals(mode=mode), 2)
        residual_sum = np.sum(squared_residuals, axis=0)
        excluded_sum = residual_sum - squared_residuals
        excluded_mse = excluded_sum / (squared_residuals.shape[0] - 1)
        jackknife_variance = np.var(
            excluded_mse, axis=0) * (squared_residuals.shape[0] - 1)

        return jackknife_variance

    def feature_remaining_error(self, truth=None, mode='additive_mean'):

        null_square_residuals = np.power(self.null_residuals(truth=truth), 2)
        null_residual_sum = np.sum(null_square_residuals, axis=0)

        forest_square_residuals = np.power(self.residuals(truth=truth), 2)
        predicted_residual_sum = np.sum(forest_square_residuals, axis=0)

        remaining = predicted_residual_sum / null_residual_sum

        return remaining

    def feature_coefficient_of_determination(self, truth=None, mode='additive_mean'):
        remaining_error = self.feature_remaining_error(truth=truth, mode=mode)
        return 1 - remaining_error

    def compare_feature_residuals(self, other, mode='rank_sum', no_plot=True):
        from scipy.stats import ranksums
        from scipy.stats import mannwhitneyu
        from scipy.stats import ks_2samp
        from scipy.stats import t

        self_residuals = self.residuals()
        other_residuals = other.residuals()

        if mode == 'rank_sum':
            results = [ranksums(self_residuals[:, i], other_residuals[:, i])
                       for i in range(self_residuals.shape[1])]
        elif mode == 'mann_whitney_u':
            results = [mannwhitneyu(self_residuals[:, i], other_residuals[:, i])
                       for i in range(self_residuals.shape[1])]
        elif mode == 'kolmogorov_smirnov':
            results = [ks_2samp(self_residuals[:, i], other_residuals[:, i])
                       for i in range(self_residuals.shape[1])]

        elif mode == 'mse_delta':

            self_mse = self.feature_mse()
            other_mse = other.feature_mse()

            delta_mse = self_mse - other_mse

            jackknife_std = np.sqrt(self.jackknife_feature_mse_variance())
            jackknife_z = delta_mse / jackknife_std

            prob = t.pdf(jackknife_z, len(self.forest.samples))

            results = list(zip(jackknife_z, prob))

        elif mode == 'cod_delta':

            print("WARNING")
            print("NO SIGNFIFICANCE SCORE IS PROVIDED FOR DIFFERENCE IN COD")

            self_cod = self.feature_coefficient_of_determination()
            other_cod = other.feature_coefficient_of_determination()

            delta_cod = self_cod - other_cod

            results = list(zip(delta_cod, np.ones(len(delta_cod))))

        else:
            raise Exception(f"Did not recognize mode:{mode}")

        if not no_plot:
            plt.figure()
            plt.title("Distribution of Test Statistics")
            plt.hist([test for test, p in results], log=True, bins=50)
            plt.xlabel("Test Statistic")
            plt.ylabel("Frequency")
            plt.show()

            plt.figure()
            plt.title("Distribution of P Values")
            plt.hist([p for test, p in results], log=True, bins=50)
            plt.xlabel("P Value")
            plt.ylabel("Frequency")
            plt.show()

        return results
