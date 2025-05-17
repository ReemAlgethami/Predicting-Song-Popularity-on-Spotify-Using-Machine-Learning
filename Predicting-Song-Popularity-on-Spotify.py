import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                           confusion_matrix, classification_report, 
                           silhouette_score)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Set environment to avoid memory leaks
os.environ['OMP_NUM_THREADS'] = '1'

# ======================
# 1. Setup Environment
# ======================

def setup_environment():
    folders = [
        'data/original_data',
        'data/preprocessed_data',
        'results/regression/models',
        'results/classification/models', 
        'results/clustering/models',
        'visualizations/eda',
        'visualizations/regression',
        'visualizations/classification',
        'visualizations/clustering',
        'report'
    ]
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"Created folder: {folder}")
        except Exception as e:
            print(f"Error creating {folder}: {str(e)}")
            sys.exit(1)

# ======================
# 2. Data Loading
# ======================

def load_data():
    """Load data with multiple fallback options"""
    possible_paths = [
        'data/original_data/spotify_tracks.csv',
        'data/original_data/spotify_top_1000_tracks.csv',
        r'C:\Users\ralfa\Downloads\spotify_tracks.csv',
        r'C:\Users\ralfa\Downloads\spotify_top_1000_tracks.csv'
    ]
    
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            print(f"Successfully loaded data from: {path}")
            
            # Clean data - remove rows with missing target values
            df = df.dropna(subset=['popularity'])

            # Convert release_date to datetime
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df = df.dropna(subset=['release_date'])
            df['release_year'] = df['release_date'].dt.year

            # Add duration in minutes
            df['duration_min'] = df['duration_min'] / (1000 * 60)

            return df
        except FileNotFoundError:
            continue

    print("Error: Could not find data file in any expected location")
    sys.exit(1)

# ======================
# 3. Data Preprocessing
# ======================

def preprocess_data(df):
    """Handle missing values, feature engineering, and splitting"""
    # Feature Engineering
    for col in ['track_name', 'artist', 'album']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

    # Select features and target
    features = ['track_name_encoded', 'artist_encoded', 'album_encoded', 
               'release_year', 'duration_min']
    X = df[features]
    y = df['popularity']
    feature_names = X.columns.tolist()

    # Create classification target
    y_class = pd.cut(y, bins=[0, 60, 75, 90, 100], 
                    labels=['Low', 'Medium', 'High', 'Very High'])

    # Remove any remaining NaN values
    valid_indices = y_class.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    y_class = y_class[valid_indices].cat.remove_unused_categories()

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Split for classification (stratified)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_clf_scaled = scaler.fit_transform(X_train_clf)
    X_test_clf_scaled = scaler.transform(X_test_clf)
#  CSV
    df.to_csv('data/preprocessed_data/cleaned_spotify_data.csv', index=False)
    pd.DataFrame(X).to_csv('data/preprocessed_data/X_features.csv', index=False)
    pd.DataFrame(y).to_csv('data/preprocessed_data/y_target.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/preprocessed_data/X_test.csv', index=False)
    pd.DataFrame(y_test).to_csv('data/preprocessed_data/y_test.csv', index=False)
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'X_train_clf': X_train_clf,
        'X_test_clf': X_test_clf,
        'y_train_clf': y_train_clf,
        'y_test_clf': y_test_clf,
        'X_train_clf_scaled': X_train_clf_scaled,
        'X_test_clf_scaled': X_test_clf_scaled,
        'df': df,
        'feature_names': feature_names
    }

# ======================
# 4. Visualization Utilities
# ======================

def save_plot(filename, folder='eda', show=False):
    """Save plot to specified folder"""
    os.makedirs(f'visualizations/{folder}', exist_ok=True)
    path = f'visualizations/{folder}/{filename}'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

# ======================
# 5. EDA Visualizations
# ======================

def generate_eda_visuals(df, X_train, y_train):
    """Generate exploratory data analysis visuals"""
    print("\nGenerating EDA visualizations...")

    # 1. Top Artists
    plt.figure(figsize=(12,6))
    top_artists = df['artist'].value_counts().head(10)
    sns.barplot(x=top_artists.values, y=top_artists.index, palette='viridis')
    plt.title('Top 10 Artists by Number of Tracks')
    plt.xlabel('Number of Tracks')
    plt.ylabel('Artist')
    save_plot('top_artists.png', 'eda')

    # 2. Popularity Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y_train, bins=30, kde=True)
    plt.title('Popularity Score Distribution')
    plt.xlabel('Popularity Score')
    plt.ylabel('Count')
    save_plot('popularity_distribution.png', 'eda')

    # 3. Release Year Trends
    plt.figure(figsize=(12, 6))
    df['release_year'].value_counts().sort_index().plot(kind='bar')
    plt.title('Tracks by Release Year')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Tracks')
    save_plot('release_years.png', 'eda')

    # 4. Feature Correlation
    plt.figure(figsize=(12, 8))
    corr = X_train.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    save_plot('correlation_matrix.png', 'eda')

# ======================
# 6. Regression Analysis
# ======================

def run_regression_analysis(X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    """Train and evaluate regression models"""
    print("\nRunning regression analysis...")

    models = [
        ('Linear_Regression', LinearRegression()),
        ('Decision_Tree', DecisionTreeRegressor(random_state=42)),
        ('Random_Forest', RandomForestRegressor(random_state=42)),
        ('KNN', KNeighborsRegressor()),
        ('Neural_Network', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=5000, random_state=42)),
        ('SVM', SVR())
    ]

    results = []
    for name, model in models:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results.append([name, mse, rmse, r2])
        
        # Save detailed predictions
        pred_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Residual': y_test - y_pred
        })
        pred_df.to_csv(f'results/regression/models/{name}_predictions.csv', index=False)
        
        # Plot predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f'{name} Predictions')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        save_plot(f'{name}_predictions.png', 'regression')

    # Save summary results
    results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'RMSE', 'R2'])
    results_df.to_csv('results/regression/regression_results.csv', index=False)

    # Plot model comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='R2', data=results_df)
    plt.title('Regression Models Comparison')
    plt.xticks(rotation=45)
    save_plot('regression_performance.png', 'regression')

    # Feature importance
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train_scaled, y_train)
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    importance.to_csv('results/regression/feature_importance.csv', index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance)
    plt.title('Feature Importance')
    save_plot('feature_importance.png', 'regression')

    return results_df

# ======================
# 7. Classification Analysis
# ======================

def run_classification_analysis(X_train_clf_scaled, X_test_clf_scaled, y_train_clf, y_test_clf):
    """Train and evaluate classification models"""
    print("\nRunning classification analysis...")

    models = [
        ('Logistic_Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Decision_Tree', DecisionTreeClassifier(random_state=42)),
        ('Random_Forest', RandomForestClassifier(random_state=42)),
        ('KNN', KNeighborsClassifier()),
        ('Naive_Bayes', GaussianNB()),
        ('SVM', SVC(probability=True, random_state=42)),
        ('Neural_Network', MLPClassifier(hidden_layer_sizes=(100,50), max_iter=2000, random_state=42))
    ]

    results = []
    for name, model in models:
        model.fit(X_train_clf_scaled, y_train_clf)
        y_pred = model.predict(X_test_clf_scaled)
        y_prob = model.predict_proba(X_test_clf_scaled) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_clf, y_pred)
        report = classification_report(y_test_clf, y_pred, output_dict=True, zero_division=0)
        f1 = report['weighted avg']['f1-score']
        results.append([name, accuracy, f1])
        
        # Save detailed predictions
        pred_df = pd.DataFrame({
            'Actual': y_test_clf,
            'Predicted': y_pred
        })
        
        if y_prob is not None:
            for i, cls in enumerate(model.classes_):
                pred_df[f'Probability_{cls}'] = y_prob[:, i]
        
        pred_df.to_csv(f'results/classification/models/{name}_predictions.csv', index=False)
        
        # Save classification report
        pd.DataFrame(report).transpose().to_csv(f'results/classification/models/{name}_report.csv')
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test_clf, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        save_plot(f'{name}_confusion_matrix.png', 'classification')

    # Save summary results
    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1'])
    results_df.to_csv('results/classification/classification_results.csv', index=False)

    # Plot model comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Accuracy', data=results_df)
    plt.title('Classification Models Comparison')
    plt.xticks(rotation=45)
    save_plot('classification_performance.png', 'classification')

    return results_df

# ======================
# 8. Clustering Analysis
# ======================

def run_clustering_analysis(X_train_scaled, feature_names):
    """Perform clustering analysis"""
    print("\nRunning clustering analysis...")

    # Test different cluster numbers
    results = []
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_train_scaled)
        silhouette_avg = silhouette_score(X_train_scaled, cluster_labels)
        results.append([n_clusters, silhouette_avg])
        
        # Save cluster assignments
        cluster_df = pd.DataFrame(X_train_scaled, columns=feature_names)
        cluster_df['Cluster'] = cluster_labels
        cluster_df.to_csv(f'results/clustering/models/kmeans_{n_clusters}_clusters.csv', index=False)

    # Find optimal clusters
    results_df = pd.DataFrame(results, columns=['n_clusters', 'silhouette_score'])
    best_k = results_df.loc[results_df['silhouette_score'].idxmax(), 'n_clusters']

    # Save silhouette scores
    results_df.to_csv('results/clustering/silhouette_scores.csv', index=False)

    # Final clustering with optimal k
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_train_scaled)

    # Save final clusters
    final_clusters = pd.DataFrame(X_train_scaled, columns=feature_names)
    final_clusters['Cluster'] = cluster_labels
    final_clusters.to_csv('results/clustering/final_clusters.csv', index=False)

    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=cluster_labels, palette='viridis')
    plt.title(f'Cluster Visualization (k={best_k})')
    save_plot('cluster_pca.png', 'clustering')

    return best_k

# ======================
# 9. Main Execution
# ======================

def main():
    # 1. Setup environment
    setup_environment()

    # 2. Load data
    df = load_data()

    # 3. Preprocess data
    data = preprocess_data(df)

    # 4. Generate EDA visuals
    generate_eda_visuals(data['df'], data['X_train'], data['y_train'])

    # 5. Run analyses
    regression_results = run_regression_analysis(
        data['X_train_scaled'], data['X_test_scaled'], 
        data['y_train'], data['y_test'],
        data['feature_names'])

    classification_results = run_classification_analysis(
        data['X_train_clf_scaled'], data['X_test_clf_scaled'],
        data['y_train_clf'], data['y_test_clf'])

    optimal_clusters = run_clustering_analysis(
        data['X_train_scaled'], data['feature_names'])

    # 6. Print summary
    print("\n=== Analysis Summary ===")
    print("\nRegression Results:")
    print(regression_results.to_string())

    print("\nClassification Results:")
    print(classification_results.to_string())

    print(f"\nOptimal number of clusters: {optimal_clusters}")
    print("\nAll results saved to CSV files in the results/ folder")
    print("All visualizations saved to visualizations/ folder")

if __name__ == "__main__":
    main()                                                                                                                                              
