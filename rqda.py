import csv
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap

def load_data_from_csv(filename, dimension=2):
    '''
    Load data with features and class labels from a CSV file
    '''
    X = []  # to store feature vectors
    y = []  # to store labels

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert the first 'dimension' elements of the row to floats (features)
            X.append([float(value) for value in row[:dimension]])
            # Store the last element (label)
            y.append(row[-1])
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def encode_labels(y, label_mapping):
    '''
    Reformat a list y of labels contained within the label mapping,
    return a new list containing their relationship in the mapping
    '''
    return np.array([label_mapping[item] for item in y])

def get_all_labels(y):
    '''
    Generate a mapping from label names to numeric indices
    '''
    labels = np.unique(y)
    return {label: idx for idx, label in enumerate(labels)}

import numpy as np
from typing import Literal

class RQDAClassifier:
    def __init__(self, covariance_type: Literal['general', 'independent', 'isotropic'] = 'general', nu=0.5):
        # Initialize dictionaries to store class-specific means, covariances, and priors
        self.means = {}
        self.covariances = {}
        self.priors = {}
        
        # Covariance type: 'general', 'independent', or 'isotropic'
        self.covariance_type = covariance_type
        
        # Regularization parameter nu for blending class-specific and pooled covariance
        self.nu = nu
        self.pooled_covariance = None
    
    def fit(self, X, y):
        # Find unique classes
        classes = np.unique(y)
        
        # Calculate pooled covariance (RQDA's covariance matrix)
        if self.covariance_type == 'general':
            self.pooled_covariance = np.cov(X, rowvar=False)
        elif self.covariance_type == 'independent':
            self.pooled_covariance = np.diag(np.var(X, axis=0))
        elif self.covariance_type == 'isotropic':
            self.pooled_covariance = np.identity(X.shape[1]) * np.mean(np.var(X, axis=0))

        # Iterate over each class
        for cls in classes:
            # Subset of data belonging to the current class
            X_cls = X[y == cls]
            
            # Calculate mean and prior probability for the class
            self.means[cls] = np.mean(X_cls, axis=0)
            self.priors[cls] = X_cls.shape[0] / X.shape[0]
            
            cov_k = None

            # Class-specific covariance
            if self.covariance_type == 'general':
                cov_k = np.cov(X_cls, rowvar=False)
            elif self.covariance_type == 'independent':
                cov_k = np.diag(np.var(X_cls, axis=0))
            elif self.covariance_type == 'isotropic':
                cov_k = np.identity(X.shape[1]) * np.mean(np.var(X_cls, axis=0))
            
            if cov_k is None:
                print('cov_k is none!')
                print(self.covariance_type)
                print(self.nu)

            # Apply regularization: blend class-specific and pooled covariance
            self.covariances[cls] = self.nu * cov_k + (1 - self.nu) * self.pooled_covariance
    
    def _compute_likelihood(self, x, mean, cov):
        # Helper function to calculate log-likelihood for a sample x
        size = len(mean)
        det_cov = np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)
        diff = x - mean
        return -0.5 * (np.log(det_cov) + diff.T @ inv_cov @ diff + size * np.log(2 * np.pi))
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            likelihoods = {}
            for cls in self.means:
                likelihoods[cls] = np.log(self.priors[cls]) + self._compute_likelihood(x, self.means[cls], self.covariances[cls])
            predictions.append(max(likelihoods, key=likelihoods.get))
        
        return np.array(predictions)



def plot_rqda_decision_boundary(classifier, X, y, title, ax):
    # Define the feature region
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    size = 200

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, size),
                         np.linspace(y_min, y_max, size))
    
    # Stack meshgrid for classifier prediction
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict class for each point in the grid
    zz = classifier.predict(grid).reshape(xx.shape)
    
    # Define colors for classes
    colors = ['red', 'green', 'blue']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Plot the decision boundary using consistent colors
    ax.contourf(xx, yy, zz, alpha=0.3, cmap=cmap)
    
    for cls in np.unique(y):
        ax.scatter(X[y == cls, 0], X[y == cls, 1], color=colors[cls], label=f"Class {cls}", alpha=0.6)
    
    # Plot mean vectors
    for cls, mean in classifier.means.items():
        ax.plot(mean[0], mean[1], marker='X', markersize=10, color=colors[cls], label=f"Mean Class {cls}")
    
    # Plot pooled covariance ellipse around each mean
    cov = classifier.pooled_covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues)
    for mean in classifier.means.values():
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor='black', facecolor='none', linestyle='--', linewidth=1)
        ax.add_patch(ell)
    
    # Final plot settings
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)

    ax.legend(loc="upper right", fontsize="small", markerscale=0.7, bbox_to_anchor=(1.05, 1))

if __name__ == '__main__':
    ratio_to_train_on = .95  # What percentage of the data should be reserved for training? (the rest are used for validation)

    filename = 'iris.csv'  # File path of data

    # Load data
    X, y = load_data_from_csv(filename)

    all_label_map = get_all_labels(y)
    y_encoded = encode_labels(y, all_label_map)

    N = len(X)

    # Assuming 3 classes with 50 samples each
    num_classes = 3
    samples_per_class = 50

    # Initialize lists to hold segmented data
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    # Loop through each class and segment based on the rules
    for class_label in range(num_classes):
        # Get indices for the current class
        print(class_label)
        class_indices = np.where(y_encoded == class_label)[0]

        # Sort indices to ensure order as in original data
        class_indices = np.sort(class_indices)
        
        # Select the segments for training, validation, and test sets
        train_indices = class_indices[:10]         # First 10 samples for training
        val_indices = class_indices[10:30]         # Next 20 samples for validation
        test_indices = class_indices[30:]          # Remaining 20 samples for testing

        # Append data to respective sets
        X_train.extend(X[train_indices])
        y_train.extend(y[train_indices])
        
        X_val.extend(X[val_indices])
        y_val.extend(y[val_indices])
        
        X_test.extend(X[test_indices])
        y_test.extend(y[test_indices])

    # Convert lists to numpy arrays for consistency
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    inverted_labels = {value: key for key, value in all_label_map.items()}

    # Encode labels
    y_train_encoded = encode_labels(y_train, all_label_map)
    # y_val_encoded = encode_labels(y_val, all_label_map)

    regularization_settings = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .45]

    for regularization_setting in regularization_settings:
        print(f'For reg_setting: {regularization_setting}')

        # Initialize and train RQDA classifier
        rqda = RQDAClassifier(nu=regularization_setting)
        rqda.fit(X_train, y_train_encoded)  # Fit the classifier to the training data

        predictions = rqda.predict(X_train)  # Run the classifier on the training data

        correct = 0
        incorrect = 0

        # Run analysis
        for input_vector, prediction, true_value in zip(X_train, predictions, y_train):
            prediction_label = inverted_labels[prediction]  # Correlate each prediction to its string label

            # print(f'For vector: {input_vector} model predicted: {prediction_label} actual result: {true_value}')

            if prediction_label == true_value:
                correct += 1
            else:
                incorrect += 1
        
        print('On training set: ')
        print(f'Model got: {correct} correct')
        print(f'Model got: {incorrect} incorrect')
        print(f'Misclassification rate: {incorrect / (incorrect + correct)}')

        predictions = rqda.predict(X_val)  # Run the classifier on the validation set 

        correct = 0
        incorrect = 0

        for input_vector, prediction, true_value in zip(X_val, predictions, y_val):
            prediction_label = inverted_labels[prediction]  # Correlate each prediction to its string label

            # print(f'For vector: {input_vector} model predicted: {prediction_label} actual result: {true_value}')

            if prediction_label == true_value:
                correct += 1
            else:
                incorrect += 1
        
        print('On validation set: ')
        print(f'Model got: {correct} correct')
        print(f'Model got: {incorrect} incorrect')
        print(f'Misclassification rate: {incorrect / (incorrect + correct)}')

        predictions = rqda.predict(X_test)  # Run the classifier on the validation set 

        correct = 0
        incorrect = 0

        for input_vector, prediction, true_value in zip(X_test, predictions, y_test):
            prediction_label = inverted_labels[prediction]  # Correlate each prediction to its string label

            # print(f'For vector: {input_vector} model predicted: {prediction_label} actual result: {true_value}')

            if prediction_label == true_value:
                correct += 1
            else:
                incorrect += 1
        
        print('On test set: ')
        print(f'Model got: {correct} correct')
        print(f'Model got: {incorrect} incorrect')
        print(f'Misclassification rate: {incorrect / (incorrect + correct)}')

    # Graph the model
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()  # Flatten the 3x3 grid for easy iteration

    for i, regularization_setting in enumerate(regularization_settings):
        rqda = RQDAClassifier(nu = regularization_setting)
        rqda.fit(X_train, y_train_encoded)
        plot_rqda_decision_boundary(rqda, X_train, y_train_encoded, f"RQDA ({regularization_setting} Regularization)", axes[i])

    plt.savefig('Figure-3-1.png')
    plt.show()