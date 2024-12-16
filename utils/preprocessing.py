use_gpu = True

# Data Manipulation
import pandas as pd
import numpy as np

# Math & Statistics
import math
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn Models & Tools
from sklearn.ensemble import IsolationForest
# To enable IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import (LabelEncoder, KBinsDiscretizer, PolynomialFeatures,
                                   PowerTransformer, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler, MinMaxScaler)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Imbalanced Learn
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

# Cuml (GPU Accelerated Libraries)
# if use_gpu:
#     from cuml.ensemble import RandomForestRegressor, RandomForestClassifier
# else:
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Scikit-learn Base Classes
from sklearn.base import BaseEstimator, TransformerMixin


def is_normal_distribution(data, alpha=0.05):
    """
    Fungsi untuk memeriksa apakah data mendekati distribusi normal menggunakan uji Shapiro-Wilk.

    Parameters:
    - data: array atau list, data yang akan diuji
    - alpha: float, tingkat signifikansi untuk menentukan apakah data normal (default 0.05)

    Returns:
    - True jika data mendekati distribusi normal, False jika tidak
    - Nilai p-value dari uji Shapiro-Wilk
    """
    # Menghapus missing values untuk uji Shapiro-Wilk
    data_clean = data.dropna()

    # Melakukan uji Shapiro-Wilk setelah menangani nilai NaN sementara
    stat, p_value = stats.shapiro(data_clean)

    # Menentukan apakah data mendekati distribusi normal atau tidak
    if p_value > alpha:
        return True
    else:
        return False


class FeatureOutliersHandling(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, contamination=0.05, random_state=42, temporary_imputation_strategy='mean', outlier_method='iforest'):
        """
        Parameters:
        - numerical_features: List of numerical features to apply outlier detection.
        - contamination: The proportion of the dataset expected to be outliers (for Isolation Forest).
        - random_state: Seed used by the random number generator.
        - temporary_imputation_strategy: Strategy to temporarily handle missing values ('mean', 'median').
        - outlier_method: The method for handling outliers ('iforest' or 'iqr-zscore').
        """
        self.numerical_features = numerical_features
        self.contamination = contamination
        self.random_state = random_state
        self.temporary_imputation_strategy = temporary_imputation_strategy
        self.outlier_method = outlier_method
        self.iforest = IsolationForest(
            contamination=self.contamination, random_state=self.random_state)
        self.outlier_labels_ = None

    def _impute_missing_values_temporarily(self, X):
        """
        Temporarily impute missing values only for the purpose of checking normality, 
        without modifying the original dataset.

        Parameters:
        - X: The input DataFrame.

        Returns:
        - X_imputed: DataFrame with missing values imputed (temporarily).
        """
        X_imputed = X.copy()

        if self.temporary_imputation_strategy == 'mean':
            X_imputed[self.numerical_features] = X[self.numerical_features].fillna(
                X[self.numerical_features].mean())
        elif self.temporary_imputation_strategy == 'median':
            X_imputed[self.numerical_features] = X[self.numerical_features].fillna(
                X[self.numerical_features].median())
        else:
            raise ValueError(
                "Imputation strategy not recognized. Use 'mean' or 'median'.")

        return X_imputed

    def _iqr_outliers(self, X, feature):
        """
        Detect outliers using the IQR method.
        """
        Q1 = X[feature].quantile(0.25)
        Q3 = X[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (X[feature] < lower_bound) | (X[feature] > upper_bound)

    def _zscore_outliers(self, X, feature):
        """
        Detect outliers using the Z-Score method.
        """
        z_scores = np.abs(stats.zscore(X[feature].dropna()))
        return z_scores > 3

    def _detect_outliers(self, X, feature):
        """
        Detect outliers based on the specified outlier_method (iforest or iqr-zscore).
        """
        if self.outlier_method == 'iforest':
            # Fit Isolation Forest on the feature with imputed data
            X_imputed = self._impute_missing_values_temporarily(X)
            self.iforest.fit(X_imputed[[feature]])
            outliers = self.iforest.predict(X_imputed[[feature]]) == -1

        elif self.outlier_method == 'iqr-zscore':
            # Check if the feature is normally distributed (impute only for normality check)
            X_imputed = self._impute_missing_values_temporarily(X)
            if is_normal_distribution(X_imputed[feature]):
                # Use Z-Score if data is normally distributed
                outliers = self._zscore_outliers(X, feature)
            else:
                # Use IQR if data is not normally distributed
                outliers = self._iqr_outliers(X, feature)

        else:
            raise ValueError(
                "Outlier method not recognized. Use 'iforest' or 'iqr-zscore'.")
        return outliers

    def fit(self, X, numerical_features_to_handle=None, y=None):
        """
        Fits the outlier detection model on specified numerical features.

        Parameters:
        - X: The input DataFrame containing the numerical features.
        - numerical_features_to_handle: Subset of numerical features to handle outliers (optional).
        """
        # If no specific numerical_features_to_handle provided, use all numerical_features
        if numerical_features_to_handle is None:
            numerical_features_to_handle = self.numerical_features

        for feature in numerical_features_to_handle:
            # Detect outliers for each feature
            self.outlier_labels_ = self._detect_outliers(X, feature)
        return self

    def transform(self, X, numerical_features_to_handle=None):
        """
        Transforms the data by converting outliers to NaN in specified numerical features only.

        Parameters:
        - X: The input DataFrame containing the numerical features.
        - numerical_features_to_handle: Subset of numerical features to handle outliers (optional).

        Returns:
        - Transformed DataFrame with outliers in specified numerical features replaced by NaN.
        """
        # If no specific numerical_features_to_handle provided, use all numerical_features
        if numerical_features_to_handle is None:
            numerical_features_to_handle = self.numerical_features

        # Copy the original data to avoid modification
        X_transformed = X.copy()

        # Detect and set outliers to NaN
        for feature in numerical_features_to_handle:
            outliers = self._detect_outliers(X, feature)
            X_transformed.loc[outliers, feature] = np.nan

        return X_transformed

    def fit_transform(self, X, numerical_features_to_handle=None, y=None):
        """
        Fit to data, then transform it by converting outliers to NaN in specified numerical features only.

        Parameters:
        - X: The input DataFrame containing the numerical features.
        - numerical_features_to_handle: Subset of numerical features to handle outliers (optional).

        Returns:
        - Transformed DataFrame with outliers in specified numerical features replaced by NaN.
        """
        return self.fit(X, numerical_features_to_handle).transform(X, numerical_features_to_handle)

    def plot_boxplots_for_numerical_features(self, X):
        """
        Create boxplots for all numerical features in the dataset, highlighting outliers using the chosen method.

        Parameters:
        - X: The input DataFrame containing the features.
        """
        # Calculate the number of rows and columns for the subplot grid
        num_features = len(self.numerical_features)
        cols = 2  # You can adjust this number of columns if needed
        # Calculate number of rows needed
        rows = math.ceil(num_features / cols)

        # Create a figure with subplots
        fig, axes = plt.subplots(
            nrows=rows, ncols=cols, figsize=(15, 5 * rows))

        # Set the overall figure title
        fig.suptitle(
            f'Boxplots untuk Fitur Numerik dengan Deteksi Outliers ({self.outlier_method})', fontsize=20, y=1)

        # Flatten axes for easy iteration
        axes = axes.flatten()

        # Iterate through each numerical feature and create boxplots
        for i, feature in enumerate(self.numerical_features):
            # Detect outliers for the feature
            outliers = self._detect_outliers(X, feature)

            # Create the boxplot
            axes[i].boxplot(X[feature].dropna(), vert=False)

            # Highlight the outliers
            outliers_values = X[feature][outliers]
            axes[i].scatter(outliers_values, np.ones(
                len(outliers_values)), color='red', label='Outliers', zorder=3)

            # Customize the plot
            axes[i].set_title(f'Feature: {feature}')
            axes[i].set_xlabel(feature)

        # Remove any extra subplots if there are fewer features than axes
        if len(self.numerical_features) < len(axes):
            for j in range(len(self.numerical_features), len(axes)):
                fig.delaxes(axes[j])

        # Adjust layout to avoid overlap
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()


class FeatureImputer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, categorical_features, int_num_features=None,
                 imputer_type='iterative', num_strategy='mean', cat_strategy='most_frequent'):
        """
        Parameters:
        - numerical_features: List of numerical columns to be imputed.
        - categorical_features: List of categorical columns to be imputed.
        - int_num_features: List of numerical features that should be treated as integers.
        - imputer_type: Choose 'iterative' or 'simple' to specify which imputer to use.
        - num_strategy: The strategy to use for numerical imputation ('mean', 'median', etc.)
                        when using SimpleImputer.
        - cat_strategy: The strategy to use for categorical imputation ('most_frequent', etc.)
                        when using SimpleImputer.
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.int_num_features = int_num_features if int_num_features else []
        self.imputer_type = imputer_type
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy

        # Select imputer for numerical features
        if self.imputer_type == 'iterative':
            self.imp_num = IterativeImputer(estimator=RandomForestRegressor(random_state=42),
                                            initial_strategy=self.num_strategy, random_state=42)
        else:
            self.imp_num = SimpleImputer(strategy=self.num_strategy)

        # Select imputer for categorical features
        if self.imputer_type == 'iterative':
            self.imp_cat = IterativeImputer(estimator=RandomForestClassifier(random_state=42),
                                            initial_strategy=self.cat_strategy, random_state=42)
        else:
            self.imp_cat = SimpleImputer(strategy=self.cat_strategy)

        self.label_encoders = {}
        self.original_dtypes = {}

    def _encode_categorical(self, X):
        """
        Converts categorical features to numerical using LabelEncoder,
        handling NaN values separately (keeping them as NaN).
        """
        X_cat_encoded = X[self.categorical_features].copy()

        for col in self.categorical_features:
            # Store original dtype
            self.original_dtypes[col] = X_cat_encoded[col].dtype
            le = LabelEncoder()

            # Handle missing values separately by preserving NaN values
            mask = pd.isna(X_cat_encoded[col])
            X_cat_encoded[col] = le.fit_transform(
                X_cat_encoded[col].astype(str))
            X_cat_encoded.loc[mask, col] = np.nan  # Restore NaN after encoding
            self.label_encoders[col] = le

        return X_cat_encoded

    def _decode_categorical(self, X_cat_encoded):
        """
        Converts encoded numerical categorical features back to original categories.
        """
        X_cat_decoded = X_cat_encoded.copy()

        for col in self.categorical_features:
            le = self.label_encoders[col]
            mask = pd.isna(X_cat_encoded[col])  # Identify missing values

            X_cat_decoded[col] = le.inverse_transform(
                X_cat_encoded[col].astype(int))
            X_cat_decoded.loc[mask, col] = np.nan  # Restore NaN after decoding

            # Convert back to original dtype
            if self.original_dtypes[col] == 'float64':
                X_cat_decoded[col] = X_cat_decoded[col].astype('float64')
            elif self.original_dtypes[col] == 'int64':
                X_cat_decoded[col] = X_cat_decoded[col].astype('int64')

        return X_cat_decoded

    def _convert_to_int(self, X):
        """
        Converts specified numerical features to integers.
        """
        X_int_converted = X.copy()

        for col in self.int_num_features:
            X_int_converted[col] = X_int_converted[col].round().astype(int)

        return X_int_converted

    def _restore_original_dtypes(self, X):
        """
        Restore the original dtypes of the numerical features if they were float before imputation.
        """
        for col in self.numerical_features:
            original_dtype = self.original_dtypes.get(col, None)
            if original_dtype == 'float64':
                X[col] = X[col].astype('float64')
            elif original_dtype == 'int64':
                X[col] = X[col].astype('int64')
        return X

    def fit(self, X, y=None):
        # Store the original data types for numerical and categorical columns
        for col in self.numerical_features + self.categorical_features:
            self.original_dtypes[col] = X[col].dtype

        # Impute numerical features
        self.imp_num.fit(X[self.numerical_features])

        # Encode and impute categorical features
        X_cat_encoded = self._encode_categorical(X)
        self.imp_cat.fit(X_cat_encoded)

        return self

    def transform(self, X):
        # Impute numerical features
        X_num = pd.DataFrame(self.imp_num.transform(X[self.numerical_features]),
                             columns=self.numerical_features, index=X.index)

        # Convert numerical features that should be integers
        X_num = self._convert_to_int(X_num)

        # Encode, impute, and decode categorical features
        X_cat_encoded = self._encode_categorical(X)
        X_cat_imputed_encoded = pd.DataFrame(self.imp_cat.transform(X_cat_encoded),
                                             columns=self.categorical_features, index=X.index)
        X_cat = self._decode_categorical(X_cat_imputed_encoded)

        # Restore the original dtypes for numerical features
        X_num = self._restore_original_dtypes(X_num)

        # Combine the numerical and categorical data
        return pd.concat([X_num, X_cat], axis=1)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class FeatureResampling(BaseEstimator, TransformerMixin):
    def __init__(self, resampling_method='undersampling', categorical_features=None, sampling_strategy='auto', random_state=42):
        """
        Parameters:
        - resampling_method: 'undersampling' or 'oversampling', method to use for resampling the dataset.
        - categorical_features: List of categorical features to be used when oversampling with SMOTENC (optional).
        - sampling_strategy: The sampling strategy for resampling (auto, float, or dict).
        - random_state: Seed for random number generator to ensure reproducibility.
        """
        self.resampling_method = resampling_method
        self.categorical_features = categorical_features
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        No fitting is required for resampling, so fit method does nothing.
        """
        return self

    def transform(self, X, y):
        """
        Apply undersampling or oversampling to the dataset.

        Parameters:
        - X: Feature dataset (pandas DataFrame or numpy array).
        - y: Target labels (pandas Series or numpy array).

        Returns:
        - X_resampled: Resampled features.
        - y_resampled: Resampled target labels.
        """
        # Ensure that X is a DataFrame for compatibility
        X_resampled = pd.DataFrame(X)

        if self.resampling_method == 'undersampling':
            # Apply RandomUnderSampler for undersampling
            undersampler = RandomUnderSampler(
                sampling_strategy=self.sampling_strategy, random_state=self.random_state)
            X_resampled, y_resampled = undersampler.fit_resample(
                X_resampled, y)

        elif self.resampling_method == 'oversampling':
            # Apply SMOTENC for oversampling, especially for categorical features
            if self.categorical_features is None:
                raise ValueError(
                    "For oversampling with SMOTENC, you must provide a list of categorical features.")

            smotenc = SMOTENC(categorical_features=self.categorical_features,
                              sampling_strategy=self.sampling_strategy, random_state=self.random_state)
            X_resampled, y_resampled = smotenc.fit_resample(X_resampled, y)

        else:
            raise ValueError(
                "Invalid resampling method. Choose either 'undersampling' or 'oversampling'.")

        return X_resampled, y_resampled

    def fit_transform(self, X, y=None):
        """
        Combine fit and transform in one step.

        Parameters:
        - X: Feature dataset.
        - y: Target labels.

        Returns:
        - Resampled X and y.
        """
        return self.fit(X, y).transform(X, y)

    def plot_class_count(self, y, title="Class Distribution"):
        """
        Plot a pie chart of the class distribution in the target labels y.

        Parameters:
        - y: Target labels (pandas Series or numpy array).
        - title: Title for the plot (default is 'Class Distribution').
        """
        class_counts = pd.Series(y).value_counts()
        class_labels = [f'{label} ({count} / {count / len(y) * 100:.2f}%)' for label,
                        count in zip(class_counts.index, class_counts.values)]

        # Plotting the pie chart with a smaller figure size
        # Adjusted the figsize to be smaller (6 inches width, 4 inches height)
        plt.figure(figsize=(6, 4))
        plt.pie(class_counts, labels=class_labels,
                autopct='%1.1f%%', startangle=90, counterclock=False)
        plt.title(title, fontsize=12)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.axis('equal')
        plt.show()


class FeatureLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Inisialisasi class LabelEncoder untuk melakukan encoding pada label (target variable).
        """
        self.encoder = LabelEncoder()

    def fit(self, y):
        """
        Fit the label encoder to the target variable.

        Parameters:
        - y: Target labels (pandas Series or numpy array).

        Returns:
        - self: Fitted transformer.
        """
        self.encoder.fit(y)
        return self

    def transform(self, y):
        """
        Transform the target labels using the fitted label encoder.

        Parameters:
        - y: Target labels (pandas Series or numpy array).

        Returns:
        - y_encoded: Encoded target labels.
        """
        return pd.Series(self.encoder.transform(y), index=y.index)

    def fit_transform(self, y):
        """
        Fit the label encoder, then transform the target labels in one step.

        Parameters:
        - y: Target labels (pandas Series or numpy array).

        Returns:
        - y_encoded: Encoded target labels.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded):
        """
        Inverse transform the encoded target labels back to original labels.

        Parameters:
        - y_encoded: Encoded target labels (pandas Series or numpy array).

        Returns:
        - y_original: Original target labels.
        """
        return pd.Series(self.encoder.inverse_transform(y_encoded), index=y_encoded.index)

    def get_encoding_map(self):
        """
        Returns a dictionary that maps the encoded values to the original labels.

        Returns:
        - encoding_map: Dictionary of {encoded_value: original_label}.
        """
        encoding_map = {idx: label for idx,
                        label in enumerate(self.encoder.classes_)}
        return encoding_map


class FeatureDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, encode='ordinal', strategy='uniform', numerical_features_to_discretize=None):
        """
        Parameters:
        - n_bins: Number of bins to discretize into (integer or list of integers for each feature).
        - encode: The encoding method to use ('ordinal', 'onehot', 'onehot-dense').
        - strategy: Strategy to define the width of the bins ('uniform', 'quantile', 'kmeans').
        - numerical_features_to_discretize: List of column names (or indices) of numerical features to discretize.
        """
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.numerical_features_to_discretize = numerical_features_to_discretize
        self.discretizer = None

    def fit(self, X, y=None):
        """
        Fit the KBinsDiscretizer on the specified numerical features in the provided data.

        Parameters:
        - X: Input data (pandas DataFrame, cuDF DataFrame, or numpy array).
        - y: Ignored, not used in this transformer.

        Returns:
        - self: Fitted transformer.
        """
        # Select numerical features to discretize
        if self.numerical_features_to_discretize is None:
            raise ValueError(
                "You must provide a list of numerical features to discretize.")

        # Subset the data to only include the selected numerical features
        X_selected = X[self.numerical_features_to_discretize]

        # Initialize the KBinsDiscretizer
        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, encode=self.encode, strategy=self.strategy)

        # Fit the discretizer to the selected numerical features
        self.discretizer.fit(X_selected)
        return self

    def transform(self, X):
        """
        Transform the specified numerical features by discretizing them using the fitted KBinsDiscretizer.

        Parameters:
        - X: Input data (pandas DataFrame, cuDF DataFrame, or numpy array).

        Returns:
        - X_transformed: DataFrame with discretized features, while non-discretized features remain unchanged.
        """
        # Check if the discretizer has been fitted
        if self.discretizer is None:
            raise RuntimeError(
                "You must fit the transformer before calling transform.")

        # Copy the input data to avoid modifying the original data
        X_transformed = X.copy()

        # Select numerical features to discretize
        X_selected = X[self.numerical_features_to_discretize]

        # Apply the transformation to the selected numerical features
        X_discretized = self.discretizer.transform(X_selected)

        # Handle sparse matrix for 'onehot' encoding
        if self.encode == 'onehot':
            X_discretized = X_discretized.toarray()  # Convert sparse matrix to dense

        # If the output has more columns (one-hot encoding), we need to add the new columns to the DataFrame
        if self.encode in ['onehot', 'onehot-dense']:
            num_bins = X_discretized.shape[1]
            new_col_names = [
                f'{self.numerical_features_to_discretize[0]}_bin_{i}' for i in range(num_bins)]
            # Create new DataFrame with the one-hot encoded values
            X_discretized_df = pd.DataFrame(
                X_discretized, columns=new_col_names, index=X.index)

            # Drop the original feature and concatenate the one-hot encoded features
            X_transformed = pd.concat([X_transformed.drop(
                columns=self.numerical_features_to_discretize), X_discretized_df], axis=1)
        else:
            # Replace the original numerical feature with the discretized one
            if X_discretized.ndim == 1 or X_discretized.shape[1] == 1:
                X_transformed[self.numerical_features_to_discretize[0]
                              ] = X_discretized.ravel()
            else:
                X_transformed[self.numerical_features_to_discretize] = X_discretized

        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it in one step.

        Parameters:
        - X: Input data (pandas DataFrame, cuDF DataFrame, or numpy array).
        - y: Ignored, not used in this transformer.

        Returns:
        - X_transformed: DataFrame with discretized features.
        """
        return self.fit(X, y).transform(X)

    def get_bin_edges(self):
        """
        Returns the edges of the bins created by the KBinsDiscretizer.

        Returns:
        - bin_edges: A dictionary containing the feature name and its corresponding bin edges.
        """
        if self.discretizer is None:
            raise RuntimeError(
                "You must fit the transformer before accessing bin edges.")

        # Collect bin edges for each feature
        bin_edges_dict = {}
        for i, feature in enumerate(self.numerical_features_to_discretize):
            bin_edges_dict[feature] = self.discretizer.bin_edges_[i]

        return bin_edges_dict


class FeatureRareCategoriesGrouping(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features=None, threshold=0.05, rare_label='Rare'):
        """
        Parameters:
        - categorical_features: List of column names or indices of categorical features to group rare categories.
        - threshold: Minimum proportion of occurrences for a category to be considered "rare". 
                     Categories with a proportion less than this threshold will be grouped.
        - rare_label: The label to assign to rare categories (default is 'Rare').
        """
        self.categorical_features = categorical_features
        self.threshold = threshold
        self.rare_label = rare_label
        self.mappings_ = {}  # To store the mapping for each feature

    def fit(self, X, y=None):
        """
        Fit the transformer by identifying rare categories in the specified categorical features.

        Parameters:
        - X: Input data (pandas DataFrame).
        - y: Ignored, not used in this transformer.

        Returns:
        - self: Fitted transformer.
        """
        if self.categorical_features is None:
            raise ValueError(
                "You must provide a list of categorical features.")

        for feature in self.categorical_features:
            # Calculate the proportion of each category
            value_counts = X[feature].value_counts(normalize=True)

            # Identify categories that fall below the threshold
            rare_categories = value_counts[value_counts < self.threshold].index

            # Store the mapping (categories to be grouped as 'Rare')
            self.mappings_[feature] = rare_categories.tolist()

        return self

    def transform(self, X):
        """
        Transform the data by grouping rare categories in the specified categorical features.

        Parameters:
        - X: Input data (pandas DataFrame).

        Returns:
        - X_transformed: DataFrame with rare categories grouped.
        """
        X_transformed = X.copy()

        # Replace rare categories with the rare_label
        for feature, rare_categories in self.mappings_.items():
            X_transformed[feature] = X_transformed[feature].apply(
                lambda x: self.rare_label if x in rare_categories else x)

        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it in one step.

        Parameters:
        - X: Input data (pandas DataFrame).
        - y: Ignored, not used in this transformer.

        Returns:
        - X_transformed: DataFrame with rare categories grouped.
        """
        return self.fit(X, y).transform(X)


class FeaturePolynomialAdder(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, interaction_only=False, include_bias=True, columns=None):
        """
        Inisialisasi class FeaturePolynomialAdder untuk menambahkan fitur polinomial.

        Parameters:
        - degree: Derajat maksimal dari fitur polinomial yang akan dihasilkan.
        - interaction_only: Jika True, hanya menghasilkan fitur interaksi antar fitur, tanpa pangkat fitur tunggal.
        - include_bias: Jika True, tambahkan kolom bias (fitur yang semua elemennya adalah 1).
        - columns: Daftar kolom numerik yang akan digunakan untuk penambahan fitur polinomial. Jika None, akan menggunakan semua kolom numerik.
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.columns = columns
        self.poly = None  # Placeholder for PolynomialFeatures object

    def fit(self, X, y=None):
        """
        Fit PolynomialFeatures ke data.

        Parameters:
        - X: DataFrame input yang berisi fitur.
        - y: Diabaikan, tidak digunakan dalam transformer ini.

        Returns:
        - self: Fitted transformer.
        """
        # Jika columns tidak disediakan, pilih semua kolom numerik
        if self.columns is None:
            self.columns = X.select_dtypes(
                include=['float64', 'int64']).columns.tolist()

        # Inisialisasi PolynomialFeatures dengan parameter yang diberikan
        self.poly = PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only, include_bias=self.include_bias)

        # Fit PolynomialFeatures ke kolom yang dipilih
        self.poly.fit(X[self.columns])

        return self

    def transform(self, X):
        """
        Transformasikan data dengan menambahkan fitur polinomial.

        Parameters:
        - X: DataFrame input yang berisi fitur.

        Returns:
        - X_transformed: DataFrame dengan fitur asli dan fitur polinomial tambahan.
        """
        # Pastikan PolynomialFeatures sudah di-fit
        if self.poly is None:
            raise RuntimeError(
                "Anda harus fit transformer sebelum memanggil transform.")

        # Pilih kolom numerik yang telah di-fit
        X_selected = X[self.columns]

        # Hasilkan fitur polinomial
        X_poly = self.poly.transform(X_selected)

        # Dapatkan nama fitur baru
        poly_feature_names = self.poly.get_feature_names_out(self.columns)

        # Buat DataFrame dengan fitur polinomial baru
        X_poly_df = pd.DataFrame(
            X_poly, columns=poly_feature_names, index=X.index)

        # Gabungkan fitur asli dengan fitur polinomial
        X_transformed = pd.concat(
            [X.drop(columns=self.columns), X_poly_df], axis=1)

        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit PolynomialFeatures ke data, lalu transformasikan dalam satu langkah.

        Parameters:
        - X: DataFrame input yang berisi fitur.
        - y: Diabaikan, tidak digunakan dalam transformer ini.

        Returns:
        - X_transformed: DataFrame dengan fitur asli dan fitur polinomial tambahan.
        """
        return self.fit(X, y).transform(X)


class FeaturePowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='yeo-johnson', standardize=True, columns=None):
        """
        Inisialisasi class FeaturePowerTransformer untuk melakukan transformasi power (Box-Cox atau Yeo-Johnson).

        Parameters:
        - method: Metode transformasi yang digunakan ('yeo-johnson' atau 'box-cox'). 
                  'yeo-johnson' dapat digunakan untuk data yang mengandung nilai negatif.
                  'box-cox' hanya untuk data yang bernilai positif.
        - standardize: Jika True, hasil transformasi akan di-normalisasi dengan mean=0 dan var=1.
        - columns: Daftar kolom numerik yang akan diterapkan transformasi power. Jika None, akan menggunakan semua kolom numerik.
        """
        self.method = method
        self.standardize = standardize
        self.columns = columns
        self.transformer = None  # Placeholder untuk PowerTransformer object

    def fit(self, X, y=None):
        """
        Fit PowerTransformer ke data.

        Parameters:
        - X: DataFrame input yang berisi fitur.
        - y: Diabaikan, tidak digunakan dalam transformer ini.

        Returns:
        - self: Fitted transformer.
        """
        # Jika columns tidak disediakan, pilih semua kolom numerik
        if self.columns is None:
            self.columns = X.select_dtypes(
                include=['float64', 'int64']).columns.tolist()

        # Inisialisasi PowerTransformer dengan parameter yang diberikan
        self.transformer = PowerTransformer(
            method=self.method, standardize=self.standardize)

        # Fit PowerTransformer ke kolom yang dipilih
        self.transformer.fit(X[self.columns])

        return self

    def transform(self, X):
        """
        Transformasikan data dengan menerapkan transformasi power.

        Parameters:
        - X: DataFrame input yang berisi fitur.

        Returns:
        - X_transformed: DataFrame dengan fitur yang telah diterapkan transformasi power.
        """
        # Pastikan PowerTransformer sudah di-fit
        if self.transformer is None:
            raise RuntimeError(
                "Anda harus fit transformer sebelum memanggil transform.")

        # Pilih kolom yang akan diterapkan transformasi
        X_selected = X[self.columns]

        # Terapkan transformasi
        X_transformed_values = self.transformer.transform(X_selected)

        # Buat DataFrame dengan hasil transformasi
        X_transformed_df = pd.DataFrame(
            X_transformed_values, columns=self.columns, index=X.index)

        # Gabungkan kembali dengan fitur yang tidak diterapkan transformasi
        X_remaining = X.drop(columns=self.columns)
        X_transformed = pd.concat([X_remaining, X_transformed_df], axis=1)

        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit PowerTransformer ke data, lalu transformasikan dalam satu langkah.

        Parameters:
        - X: DataFrame input yang berisi fitur.
        - y: Diabaikan, tidak digunakan dalam transformer ini.

        Returns:
        - X_transformed: DataFrame dengan fitur yang telah diterapkan transformasi power.
        """
        return self.fit(X, y).transform(X)

    def get_lambdas(self):
        """
        Mengembalikan nilai lambda yang digunakan dalam transformasi untuk setiap kolom.

        Returns:
        - lambdas_: List lambda yang digunakan oleh PowerTransformer.
        """
        if self.transformer is None:
            raise RuntimeError(
                "Anda harus fit transformer sebelum memanggil get_lambdas.")

        return dict(zip(self.columns, self.transformer.lambdas_))

    def plot_kde_hist_before_after(self, X_before, X_after):
        """
        Plot KDE dan histogram sebelum dan sesudah power transform.

        Parameters:
        - X_before: DataFrame sebelum transformasi power.
        - X_after: DataFrame setelah transformasi power.
        """
        num_cols = len(self.columns)
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))

        for i, col in enumerate(self.columns):
            # Plot distribusi sebelum transformasi
            sns.histplot(X_before[col], kde=True, ax=axes[i, 0], color='blue')
            axes[i, 0].set_title(f'{col} - Sebelum Transformasi')
            axes[i, 0].set_xlabel(col)
            axes[i, 0].set_ylabel('Density')

            # Plot distribusi setelah transformasi
            sns.histplot(X_after[col], kde=True, ax=axes[i, 1], color='green')
            axes[i, 1].set_title(f'{col} - Setelah Transformasi')
            axes[i, 1].set_xlabel(col)
            axes[i, 1].set_ylabel('Density')

        plt.tight_layout()
        plt.show()


class FeatureGroupingNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features_to_grouping, aggregations=['min', 'max', 'mean', 'median', 'std'], columns_name='BILL_AMT'):
        """
        Parameters:
        - numerical_features_to_grouping: List of numerical features that you want to group.
        - aggregations: List of aggregation functions to apply (default=['min', 'max', 'mean', 'median', 'std']).
        - columns_name: Prefix for the new grouped feature names.
        """
        self.numerical_features_to_grouping = numerical_features_to_grouping
        self.aggregations = aggregations
        self.columns_name = columns_name

    def fit(self, X, y=None):
        """
        No fitting required for this transformer, so fit method does nothing.
        """
        return self

    def transform(self, X):
        """
        Apply aggregation on the selected numerical features and return a DataFrame with the aggregated features.

        Parameters:
        - X: DataFrame containing the input data.

        Returns:
        - X_transformed: DataFrame with the aggregated features.
        """
        # Select only the columns that will be grouped
        X_group = X[self.numerical_features_to_grouping]

        # Initialize an empty DataFrame to store the results
        X_aggregated = pd.DataFrame(index=X.index)

        # Apply the aggregation functions
        for agg_func in self.aggregations:
            # For each aggregation function, calculate the result and name the columns
            agg_result = X_group.aggregate(agg_func, axis=1)
            column_name = f'{self.columns_name}_{agg_func}'
            X_aggregated[column_name] = agg_result

        # Combine the original data with the aggregated results
        X_transformed = pd.concat([X, X_aggregated], axis=1)

        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Combine fit and transform in one step.

        Parameters:
        - X: DataFrame containing the input data.
        - y: Ignored, not used in this transformer.

        Returns:
        - X_transformed: DataFrame with the aggregated features.
        """
        return self.fit(X, y).transform(X)

class FeatureDimensionReducer(BaseEstimator, TransformerMixin):
    def __init__(self, method='pca', n_components=None, numeric_features_to_reduce=None, column_names='component'):
        """
        Parameters:
        - method: 'pca' for Principal Component Analysis or 'lda' for Linear Discriminant Analysis.
        - n_components: Number of components to keep. For 'pca', this is the number of principal components.
                        For 'lda', this should be less than or equal to (n_classes - 1).
        - numeric_features_to_reduce: List of numeric features to apply dimensionality reduction to.
        """
        self.method = method
        self.n_components = n_components
        self.numeric_features_to_reduce = numeric_features_to_reduce
        self.column_names = column_names
        self.reducer = None
        self.variance_ratio_ = None  # Store variance ratio (for PCA)
        self.cumulative_variance_ratio_ = None  # Cumulative variance ratio for PCA

    def fit(self, X, y=None, threshold=0.90):
        """
        Fit the dimensionality reduction model (PCA or LDA) to the data.

        Parameters:
        - X: Input data (features).
        - y: Target labels (only used if 'lda' method is selected).
        - threshold: The minimum cumulative variance ratio required for PCA.

        Returns:
        - self: Fitted transformer.
        """
        if self.numeric_features_to_reduce is None:
            raise ValueError(
                "You must provide a list of numeric features to reduce.")

        # Extract the numeric features to reduce
        X_selected = X[self.numeric_features_to_reduce]

        if self.method == 'pca':
            # Perform PCA to calculate explained variance for all components
            temp_pca = PCA()
            temp_pca.fit(X_selected)

            # Calculate cumulative variance ratio
            self.variance_ratio_ = temp_pca.explained_variance_ratio_
            self.cumulative_variance_ratio_ = np.cumsum(self.variance_ratio_)

            # Select the minimum number of components to meet the threshold
            if self.n_components is None:
                self.n_components = np.argmax(self.cumulative_variance_ratio_ >= threshold) + 1

            # Print variance ratios for transparency
            print(f"Variance Ratio for each component: {self.variance_ratio_}")
            print(f"Cumulative Variance Ratio: {self.cumulative_variance_ratio_}")
            print(f"Selected number of components (threshold={threshold * 100}%): {self.n_components}")

            # Fit PCA with the selected number of components
            self.reducer = PCA(n_components=self.n_components)
            self.reducer.fit(X_selected)

        elif self.method == 'lda':
            # Use LDA for dimensionality reduction, requires target labels (supervised method)
            if y is None:
                raise ValueError("y (target labels) must be provided for LDA.")
            self.reducer = LDA(n_components=self.n_components)
            self.reducer.fit(X_selected, y)
        else:
            raise ValueError("Invalid method. Choose 'pca' or 'lda'.")

        return self

    def transform(self, X):
        """
        Apply the dimensionality reduction model to the data.

        Parameters:
        - X: Input data to be transformed.

        Returns:
        - X_transformed: Data with original non-reduced features and transformed reduced features.
        """
        if self.reducer is None:
            raise RuntimeError(
                "You must fit the transformer before calling transform.")

        # Extract numeric features to reduce
        X_selected = X[self.numeric_features_to_reduce]

        # Apply dimensionality reduction
        X_reduced = self.reducer.transform(X_selected)

        # Convert reduced features to DataFrame with appropriate column names
        reduced_feature_names = [
            f'{self.column_names}_{i+1}' for i in range(X_reduced.shape[1])]
        X_reduced_df = pd.DataFrame(
            X_reduced, columns=reduced_feature_names, index=X.index)

        # Drop original numeric features to reduce from X
        X_remaining = X.drop(columns=self.numeric_features_to_reduce)

        # Concatenate remaining features and reduced features
        X_transformed = pd.concat([X_remaining, X_reduced_df], axis=1)

        return X_transformed

    def fit_transform(self, X, y=None, threshold=0.90):
        """
        Fit to data, then transform it in one step.

        Parameters:
        - X: Input data.
        - y: Target labels (only used if 'lda' method is selected).
        - threshold: The minimum cumulative variance ratio required for PCA.

        Returns:
        - X_transformed: Data with original non-reduced features and transformed reduced features.
        """
        return self.fit(X, y, threshold).transform(X)

    def get_variance_ratio(self):
        """
        Get the variance explained by each principal component (for PCA).

        Returns:
        - variance_ratio_: Explained variance ratio for each component.
        """
        if self.method == 'pca' and self.variance_ratio_ is not None:
            return self.variance_ratio_
        else:
            raise RuntimeError(
                "Variance ratio is only available for PCA after fitting the model.")


class FeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_features_dict=None, one_hot_features=None):
        """
        Parameters:
        - ordinal_features_dict: Dictionary mapping feature names to a list of ordered categories for ordinal encoding.
        - one_hot_features: List of categorical features to be one-hot encoded.
        """
        self.ordinal_features_dict = ordinal_features_dict
        self.one_hot_features = one_hot_features
        self.ordinal_encoder = None
        self.onehot_encoder = None

    def fit(self, X, y=None):
        """
        Fit the encoders to the data.

        Parameters:
        - X: Input DataFrame with categorical and ordinal features.
        - y: Ignored, not used in this transformer.

        Returns:
        - self: Fitted transformer.
        """
        # Fit OrdinalEncoder for ordinal features
        if self.ordinal_features_dict is not None:
            self.ordinal_encoder = OrdinalEncoder(
                categories=[self.ordinal_features_dict[feature]
                            for feature in self.ordinal_features_dict]
            )
            self.ordinal_encoder.fit(
                X[list(self.ordinal_features_dict.keys())])

        # Fit OneHotEncoder for nominal categorical features
        if self.one_hot_features is not None:
            # Use sparse_output=False in place of sparse=False
            self.onehot_encoder = OneHotEncoder(
                sparse_output=False, drop='first')
            self.onehot_encoder.fit(X[self.one_hot_features])

        return self

    def transform(self, X):
        """
        Transform the data by applying the fitted encoders.

        Parameters:
        - X: Input DataFrame with categorical and ordinal features.

        Returns:
        - X_transformed: DataFrame with ordinal and one-hot encoded features.
        """
        X_transformed = X.copy()

        # Apply OrdinalEncoder for ordinal features
        if self.ordinal_features_dict is not None:
            X_ordinal_encoded = self.ordinal_encoder.transform(
                X[list(self.ordinal_features_dict.keys())])
            X_ordinal_encoded_df = pd.DataFrame(
                X_ordinal_encoded, columns=self.ordinal_features_dict.keys(), index=X.index)
            X_transformed = X_transformed.drop(
                columns=list(self.ordinal_features_dict.keys()))
            X_transformed = pd.concat(
                [X_transformed, X_ordinal_encoded_df], axis=1)

        # Apply OneHotEncoder for nominal categorical features
        if self.one_hot_features is not None:
            X_onehot_encoded = self.onehot_encoder.transform(
                X[self.one_hot_features])
            onehot_encoded_columns = self.onehot_encoder.get_feature_names_out(
                self.one_hot_features)
            X_onehot_encoded_df = pd.DataFrame(
                X_onehot_encoded, columns=onehot_encoded_columns, index=X.index)
            X_transformed = X_transformed.drop(columns=self.one_hot_features)
            X_transformed = pd.concat(
                [X_transformed, X_onehot_encoded_df], axis=1)

        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit the encoders, then transform the data.

        Parameters:
        - X: Input DataFrame with categorical and ordinal features.
        - y: Ignored, not used in this transformer.

        Returns:
        - X_transformed: DataFrame with ordinal and one-hot encoded features.
        """
        return self.fit(X, y).transform(X)


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Inisialisasi class FeatureScaler.
        """
        self.scalers = {}
        self.normal_features = []
        self.non_normal_features = []

    def fit(self, X, y=None):
        """
        Fit the scalers to the data.

        Parameters:
        - X: Input DataFrame with numeric features.
        - y: Ignored, not used in this transformer.

        Returns:
        - self: Fitted transformer.
        """
        # Mendapatkan semua kolom numerik
        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

        for feature in numeric_columns:
            # Pengecekan distribusi normal
            if is_normal_distribution(X[feature]):
                # Gunakan StandardScaler untuk distribusi normal
                scaler = StandardScaler()
                self.normal_features.append(feature)
            else:
                # Gunakan MinMaxScaler untuk yang tidak normal
                scaler = MinMaxScaler()
                self.non_normal_features.append(feature)
            # Fit scaler
            scaler.fit(X[[feature]])
            self.scalers[feature] = scaler
        
        print(f"Normal features (StandardScaler): {self.normal_features}")
        print(f"Non-normal features (MinMaxScaler): {self.non_normal_features}")

        return self

    def transform(self, X):
        """
        Transform the data by applying the fitted scalers.

        Parameters:
        - X: Input DataFrame with numeric features.

        Returns:
        - X_scaled: DataFrame with scaled numeric features.
        """
        X_scaled = X.copy()

        # Mendapatkan semua kolom numerik
        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

        for feature in numeric_columns:
            if feature in self.scalers:
                scaler = self.scalers[feature]
                X_scaled[[feature]] = scaler.transform(X[[feature]])

        return X_scaled

    def fit_transform(self, X, y=None):
        """
        Fit the scalers, then transform the data.

        Parameters:
        - X: Input DataFrame with numeric features.
        - y: Ignored, not used in this transformer.

        Returns:
        - X_scaled: DataFrame with scaled numeric features.
        """
        return self.fit(X, y).transform(X)

    def get_scaler_columns(self):
        """
        Get the lists of columns that were scaled using StandardScaler and MinMaxScaler.

        Returns:
        - normal_features: List of features scaled using StandardScaler.
        - non_normal_features: List of features scaled using MinMaxScaler.
        """
        return self.normal_features, self.non_normal_features
