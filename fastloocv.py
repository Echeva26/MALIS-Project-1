import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class FastLOOCV:
    """
    A class to perform Fast Leave-One-Out Cross-Validation (LOOCV).

    Attributes
    ----------
    data : The dataset used for performing LOOCV. Typically this includes features
           and labels required for model training and evaluation.
    """

    def __init__(self, data):
        """
        Initialize the FastLOOCV object.

        Parameters
        ----------
        data : array-like
            Input dataset to be used for LOOCV.
        """
        self.data = data

    def do_fast_loocv(self, k_values, sample_size=None):
        """
        Perform fast leave-one-out cross-validation for a set of K values.

        Parameters
        ----------
        k_values : array-like
            List or array of integer K values to evaluate (e.g., number of neighbors).
        sample_size : int, optional
            Number of samples to randomly select from the training set.
            If None, all available samples are used.

        Returns
        -------
        score : numpy.ndarray
            Vector containing performance scores for each K in k_values.
        elapsed_time : float
            Execution time (in seconds) for running the procedure.
        """
        start_time = time.time()

        # Get features (X) and target values (y) from the data
        X, y = self.data

        # If sample_size is specified and smaller than the dataset,
        # select a subset of the data
        if sample_size and sample_size < len(X):
            # For replication purposes, we take the first n samples
            # In a real scenario, you might want to use random sampling
            X_sample, y_sample = X[:sample_size], y[:sample_size]
        else:
            X_sample, y_sample = X, y
        
        scores = []

        # Iterate through each k value to evaluate
        for k in k_values:
            # Handle edge case: k=0 would cause division by zero
            if k == 0:
                scores.append(np.inf)
                continue

            # Step 1: Train a (k+1)-NN regressor using all sample data
            # Using kd_tree algorithm for efficient nearest neighbor search
            model_k_plus_1 = KNeighborsRegressor(n_neighbors=k + 1, algorithm='kd_tree')
            model_k_plus_1.fit(X_sample, y_sample)
            
            # Step 2: Make predictions on the training data
            # These predictions will be used to estimate LOOCV error
            y_pred = model_k_plus_1.predict(X_sample)
            
            # Step 3: Calculate Mean Squared Error (MSE) on training data
            # This is the raw error before applying the LOOCV scaling factor
            mse_train = np.mean((y_sample - y_pred) ** 2)
            
            # Step 4: Apply scaling factor to get LOOCV score
            # The ((k+1)/k)^2 factor adjusts for the bias in training error
            # This gives us an estimate of true LOOCV error without running n models
            scaling_factor = ((k + 1) / k) ** 2
            loocv_score = mse_train * scaling_factor
            
            scores.append(loocv_score)
        elapsed_time = time.time() - start_time
        return np.array(scores), elapsed_time
        
    def do_normal_loocv(self, k_values, sample_size=None):
        """
        Perform standard leave-one-out cross-validation for a set of K values.

        Parameters
        ----------
        k_values : array-like
            List or array of integer K values to evaluate (e.g., number of neighbors).
        sample_size : int, optional
            Number of samples to randomly select from the training set.
            If None, all available samples are used.

        Returns
        -------
        score : numpy.ndarray
            Vector containing performance scores for each K in k_values.
        elapsed_time : float
            Execution time (in seconds) for running the procedure.
        """
        start_time = time.time()

        # Placeholder: here you would implement the fast LOOCV procedure.
        # Get features and targets from the data
        X, y = self.data

        # Handle sample_size if specified
        if sample_size and sample_size < len(X):
            X_sample, y_sample = X[:sample_size], y[:sample_size]
        else:
            X_sample, y_sample = X, y
        
        n = len(X_sample)
        # Initialize score array
        scores = []
        # Iterate through each k value to evaluat
        for k in k_values:
            # Handle edge case: k=0 would cause division by zero
            if k == 0:
                scores.append(np.inf)
                continue

            mse_loocv = 0.0
            # Perform LOOCV
            for i in range(n):
                # Create training set excluding the i-th sample
                X_train = np.delete(X_sample, i, axis=0)
                y_train = np.delete(y_sample, i, axis=0)

                # Create test set with the i-th sample
                X_test = X_sample[i].reshape(1, -1)
                y_test = y_sample[i]

                # Train k-NN regressor
                model = KNeighborsRegressor(n_neighbors=k, algorithm='kd_tree')
                model.fit(X_train, y_train)

                # Predict the left-out sample
                y_pred = model.predict(X_test)

                # Accumulate squared error
                mse_loocv += (y_test - y_pred) ** 2

            # Average MSE over all samples
            mse_loocv /= n
            scores.append(mse_loocv)

        score = np.array(scores)

        elapsed_time = time.time() - start_time
        return score, elapsed_time
