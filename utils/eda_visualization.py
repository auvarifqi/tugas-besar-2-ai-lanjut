import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


def plot_pie_chart(df, label):
    """
    This function plots a pie chart for the class distribution in the label column
    and prints the value counts for each class in a nicely formatted way, 
    without modifying the original DataFrame.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    label (str): The column name for the label/class to be plotted.
    """
    # Replace NaN with 'NaN' for plotting only, without modifying the original df
    plot_data = df[label].fillna('NaN')

    # Get the value counts of the label
    class_counts = plot_data.value_counts()  # NaN values are replaced with 'Missing'
    total = len(plot_data)

    # Print the value counts with percentages in a more readable format
    print(f"Value Counts for '{label}':")
    for cls, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"  {cls}: {count} ({percentage:.2f}%)")
    
    # Plot the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(class_counts, labels=[f'{cls} ({count})' for cls, count in class_counts.items()],
            autopct=lambda p: f'{p:.2f}%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title(f"Class Distribution for '{label}'")
    plt.axis('equal')  # Ensures the pie chart is a circle
    plt.show()

def plot_boxplot(data, numerical_features, title, label=None):
    """
    Fungsi untuk memplot boxplot menggunakan sns.boxplot
    Args:
    - data: DataFrame yang berisi data
    - numerical_features: Daftar nama fitur numerik yang ingin diplot
    - title: Judul dari plot

    Output:
    - Plot boxplot untuk setiap fitur numerik
    """

    if label:
        numerical_features = numerical_features.copy()
        numerical_features.append(label)

    plt.figure(figsize=(20, 40))
    
    # Looping untuk setiap fitur numerik
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(16, 3, i)  # Mengatur layout menjadi (7, 2)
        sns.boxplot(x=data[feature])
        plt.title(f'Boxplot of {feature}')
        plt.grid(True)
    
    plt.suptitle(title, fontsize=20, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def visualize_clusters(df, labels=None, centroids=None, title="Cluster Visualization", dimensions=2):
    """
    Visualizes the dataset with clusters before or after KMeans clustering.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data (PCA-reduced features).
    - labels (np.ndarray, optional): The cluster labels for each data point. Default is None.
    - centroids (np.ndarray, optional): The centroids of the clusters. Default is None.
    - title (str): The title of the plot. Default is "Cluster Visualization".
    - dimensions (int): Whether to plot in 2D or 3D (choose 2 or 3). Default is 2.
    
    Returns:
    - None. Displays a scatter plot (2D or 3D based on input dimensions).
    """
    cluster_patches = []  # For legend
    
    # Use a vibrant colormap for clusters
    colormap = plt.cm.get_cmap('tab20', len(np.unique(labels)) if labels is not None else 20)
    
    if dimensions == 3 and df.shape[1] >= 3:
        # 3D Visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            # Color points by cluster label and create patches for legend
            scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2],
                                 c=labels, cmap=colormap, s=100, alpha=0.9, edgecolors='k')
            
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                cluster_patches.append(mpatches.Patch(color=colormap(i), 
                                                      label=f'Cluster {label}'))
        else:
            # Scatter without labels
            scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2],
                                 color='gray', s=100, alpha=0.8, edgecolors='k')
        
        if centroids is not None:
            # Plot centroids
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       s=400, c='red', marker='X', edgecolors='k', label='Centroids')
            cluster_patches.append(mpatches.Patch(color='red', label='Centroids'))
        
        # Add labels and title
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Principal Component 1", fontsize=12, fontweight='bold')
        ax.set_ylabel("Principal Component 2", fontsize=12, fontweight='bold')
        ax.set_zlabel("Principal Component 3", fontsize=12, fontweight='bold')
        ax.legend(handles=cluster_patches, loc='best', fontsize=10)
        plt.show()
    
    else:
        # 2D Visualization
        plt.figure(figsize=(12, 9))
        
        if labels is not None:
            # Color points by cluster label and create patches for legend
            scatter = plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap=colormap, 
                                  s=100, alpha=0.9, edgecolors='k')
            
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                cluster_patches.append(mpatches.Patch(color=colormap(i), 
                                                      label=f'Cluster {label}'))
        else:
            # Scatter without labels
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1], color='gray', s=100, alpha=0.8, edgecolors='k')
        
        if centroids is not None:
            # Plot centroids
            plt.scatter(centroids[:, 0], centroids[:, 1], s=400, c='red', marker='X', 
                        edgecolors='k', label='Centroids')
            cluster_patches.append(mpatches.Patch(color='red', label='Centroids'))
        
        # Add labels and title
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Principal Component 1", fontsize=12, fontweight='bold')
        plt.ylabel("Principal Component 2", fontsize=12, fontweight='bold')
        plt.legend(handles=cluster_patches, loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.8)
        plt.show()

    
def visualize_clusters_interactive(df, labels=None, centroids=None, title="3D Cluster Visualization"):
    """
    Visualizes the dataset with clusters interactively using Plotly.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data (PCA-reduced features).
    - labels (np.ndarray, optional): The cluster labels for each data point. Default is None.
    - centroids (np.ndarray, optional): The centroids of the clusters. Default is None.
    - title (str): The title of the plot. Default is "3D Cluster Visualization".

    Returns:
    - None. Displays an interactive 3D scatter plot.
    """
    # Prepare the scatter plot for data points
    scatter_points = []
    unique_labels = np.unique(labels) if labels is not None else [None]
    
    for label in unique_labels:
        # Filter data points by cluster label
        if label is not None:
            cluster_data = df[labels == label]
            cluster_name = f"Cluster {label}"
        else:
            cluster_data = df
            cluster_name = "Data Points"
        
        scatter_points.append(
            go.Scatter3d(
                x=cluster_data.iloc[:, 0],
                y=cluster_data.iloc[:, 1],
                z=cluster_data.iloc[:, 2],
                mode='markers',
                marker=dict(size=5),
                name=cluster_name
            )
        )

    # Prepare the scatter plot for centroids
    if centroids is not None:
        scatter_points.append(
            go.Scatter3d(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                mode='markers',
                marker=dict(size=10, color='red', symbol='x'),
                name='Centroids'
            )
        )

    # Create the figure
    fig = go.Figure(data=scatter_points)
    
    # Add layout configurations
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the interactive plot
    fig.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot a confusion matrix as a heatmap with counts and percentages.
    Args:
        y_true: Array-like of true labels.
        y_pred: Array-like of predicted labels.
        class_names: List of class names.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100  # Calculate percentages

    # Combine counts and percentages into one display
    cm_display = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cm_display[i, j] = f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)"

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=cm_display, fmt="", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("Confusion Matrix with Counts and Percentages", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True Labels", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()



def analyze_cluster_characteristics(df, feature_columns, cluster_col='cluster_label'):
    """
    Analyze characteristics for each cluster with formatted output
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing features and cluster assignments
    feature_columns : list
        List of feature column names to analyze
    cluster_col : str
        Name of the cluster column
    """
    # Calculate size of each cluster
    cluster_sizes = df[cluster_col].value_counts().sort_index()
    cluster_percentages = (cluster_sizes / len(df) * 100).round(2)
    
    # Create a dictionary to store cluster statistics
    cluster_summaries = {}
    
    # Calculate statistics for each cluster separately
    for cluster in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster][feature_columns]
        
        print(f"\nCluster {cluster}: ")
        print(f"Size: {cluster_sizes[cluster]} samples ({cluster_percentages[cluster]}%)")
        print("mean =")
        print(cluster_data.mean().round(2))
        print("\nmin =")
        print(cluster_data.min().round(2))
        print("\nmax =")
        print(cluster_data.max().round(2))
        print("\nmedian =")
        print(cluster_data.median().round(2))
        print("\nstd =")
        print(cluster_data.std().round(2))
        print("-" * 50)
        
        # Store statistics in dictionary if needed for later use
        cluster_summaries[cluster] = {
            'mean': cluster_data.mean(),
            'min': cluster_data.min(),
            'max': cluster_data.max(),
            'median': cluster_data.median(),
            'std': cluster_data.std()
        }
    
    return cluster_summaries

def plot_feature_distributions(df, feature_columns, cluster_col='cluster_label'):
    n_features = len(feature_columns)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 5*n_features))
    
    if n_features == 1:
        axes = [axes]
    
    for ax, feature in zip(axes, feature_columns):
        sns.boxplot(data=df, x=cluster_col, y=feature, ax=ax)
        ax.set_title(f'Distribution of {feature} across clusters')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(df, feature_columns, cluster_col='cluster_label'):
    # Calculate mean values for each feature in each cluster
    cluster_means = df.groupby(cluster_col)[feature_columns].mean()
    
    # Calculate global means
    global_means = df[feature_columns].mean()
    
    # Calculate relative differences from global mean
    relative_importance = (cluster_means - global_means) / global_means * 100
    
    plt.figure(figsize=(24, 16))
    sns.heatmap(relative_importance, 
                annot=True, 
                fmt='.1f', 
                cmap='RdYlBu_r',
                center=0,
                cbar_kws={'label': 'Relative difference from global mean (%)'})
    plt.title('Feature Importance by Cluster')
    plt.ylabel('Cluster')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.show()

def get_cluster_profiles(df, feature_columns, cluster_col='cluster_label', n_top_features=3):
    cluster_means = df.groupby(cluster_col)[feature_columns].mean()
    global_means = df[feature_columns].mean()
    relative_diff = (cluster_means - global_means) / global_means * 100
    
    profiles = {}
    for cluster in df[cluster_col].unique():
        # Get top distinguishing features
        cluster_diff = relative_diff.loc[cluster].sort_values(ascending=False)
        top_high = cluster_diff.head(n_top_features)
        top_low = cluster_diff.tail(n_top_features)
        
        profile = f"Cluster {cluster} Profile:\n"
        profile += f"Size: {len(df[df[cluster_col] == cluster])} samples\n"
        profile += "Distinguished by (percentage above average):\n"
        
        # Add high features
        profile += "Highest by consumption:\n"
        for feat, diff in top_high.items():
            profile += f"- {feat}: {diff:.1f}%\n"
        
        # Add low features
        profile += "\nLowest by consumption:\n"
        for feat, diff in top_low.items():
            profile += f"- {feat}: {diff:.1f}%\n"
            
        profiles[cluster] = profile
    
    return profiles


def plot_cluster_distribution(df, title="Distribution of Clusters"):
    plt.figure(figsize=(10, 6))
    # Get cluster counts
    cluster_counts = df['cluster_label'].value_counts().sort_index()
    
    # Create bar plot with explicit positions
    bars = plt.bar(
        range(len(cluster_counts)),
        cluster_counts,
        width=0.6,  # Make bars thinner
        color='steelblue'
    )
    
    # Customize the plot
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Dynamically set x-axis ticks and labels
    cluster_labels = cluster_counts.index.astype(str)  # Convert cluster indices to strings
    plt.xticks(range(len(cluster_counts)), cluster_labels)
    
    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight="bold"
        )
    
    # Add grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Set y-axis to start at 0
    plt.ylim(0, max(cluster_counts) * 1.1)
    
    # Add some padding to x-axis
    plt.xlim(-0.5, len(cluster_counts) - 0.5)
    
    plt.show()
