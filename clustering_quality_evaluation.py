"""
Optimized Clustering Quality Evaluation Script
Uses sklearn's built-in metrics with original embeddings for accurate and fast evaluation.

This version is 10-100x faster than the manual implementation.
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

# Import embedding generation from hdbscan_improved
try:
    from hdbscan_improved import (
        generate_embeddings,
        extract_unique_urls_with_stats,
        advanced_url_tokenization
    )
    EMBEDDING_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Could not import from hdbscan_improved.py")
    print("Make sure hdbscan_improved.py is in the same directory.")
    EMBEDDING_AVAILABLE = False


class OptimizedClusteringEvaluator:
    def __init__(self, csv_file_path, embeddings=None):
        """
        Initialize the evaluator with clustering results and embeddings.
        
        Args:
            csv_file_path (str): Path to the clustering results CSV
            embeddings (np.ndarray, optional): Pre-computed embeddings
        """
        self.csv_file_path = csv_file_path
        self.embeddings = embeddings
        self.data = []
        self.labels = None
        self.unique_urls = []
        self.load_data()
        
    def load_data(self):
        """Load clustering data from CSV file."""
        print("Loading clustering data...")
        df = pd.read_csv(self.csv_file_path, encoding='utf-8')
        
        self.unique_urls = df['masked'].tolist()
        self.labels = df['cluster'].values
        
        print(f"\nLoaded {len(self.unique_urls)} data points")
        
        # Print cluster statistics
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = np.sum(self.labels == -1)
        n_clustered = len(self.labels) - n_noise
        
        print(f"  - Clusters found: {n_clusters}")
        print(f"  - Noise points: {n_noise} ({n_noise/len(self.labels)*100:.1f}%)")
        print(f"  - Clustered points: {n_clustered} ({n_clustered/len(self.labels)*100:.1f}%)")
        
    def generate_or_load_embeddings(self, embedding_file=None, batch_size=32):
        """
        Generate embeddings or load from file.
        
        Args:
            embedding_file (str, optional): Path to saved embeddings (.npy)
            batch_size (int): Batch size for embedding generation
        """
        if self.embeddings is not None:
            print("Using provided embeddings")
            return
            
        if embedding_file and os.path.exists(embedding_file):
            print(f"Loading embeddings from {embedding_file}...")
            self.embeddings = np.load(embedding_file)
            print(f"Loaded embeddings: shape {self.embeddings.shape}")
            return
            
        if not EMBEDDING_AVAILABLE:
            raise RuntimeError(
                "Cannot generate embeddings: hdbscan_improved.py not available.\n"
                "Please provide embeddings via --embeddings parameter."
            )
            
        print("\nGenerating embeddings...")
        self.embeddings = generate_embeddings(
            self.unique_urls,
            batch_size=batch_size,
            use_pca=True
        )
        
        # Save embeddings for future use
        if embedding_file:
            np.save(embedding_file, self.embeddings)
            print(f"Saved embeddings to {embedding_file}")
    
    def calculate_silhouette_score(self):
        """
        Calculate Silhouette Score using sklearn.
        
        Returns:
            float: Silhouette score (-1 to 1, higher is better)
        """
        print("\nCalculating Silhouette Score...")
        
        # Filter out noise points
        mask = self.labels != -1
        valid_embeddings = self.embeddings[mask]
        valid_labels = self.labels[mask]
        
        if len(set(valid_labels)) < 2:
            print("Not enough clusters (need at least 2)")
            return None
            
        # Use sklearn's optimized implementation
        score = silhouette_score(
            valid_embeddings,
            valid_labels,
            metric='cosine'
        )
        
        print(f"Silhouette Score: {score:.4f}")
        return score
    
    def calculate_davies_bouldin_index(self):
        """
        Calculate Davies-Bouldin Index using sklearn.
        
        Returns:
            float: DBI (0 to inf, lower is better)
        """
        print("\nCalculating Davies-Bouldin Index...")
        
        # Filter out noise points
        mask = self.labels != -1
        valid_embeddings = self.embeddings[mask]
        valid_labels = self.labels[mask]
        
        if len(set(valid_labels)) < 2:
            print("Not enough clusters (need at least 2)")
            return None
            
        # Use sklearn's optimized implementation
        score = davies_bouldin_score(
            valid_embeddings,
            valid_labels
        )
        
        print(f"Davies-Bouldin Index: {score:.4f}")
        return score
    
    def calculate_calinski_harabasz_score(self):
        """
        Calculate Calinski-Harabasz Score (Variance Ratio Criterion).
        
        Returns:
            float: CH score (higher is better)
        """
        print("\nCalculating Calinski-Harabasz Score...")
        
        # Filter out noise points
        mask = self.labels != -1
        valid_embeddings = self.embeddings[mask]
        valid_labels = self.labels[mask]
        
        if len(set(valid_labels)) < 2:
            print("Not enough clusters (need at least 2)")
            return None
            
        # Use sklearn's optimized implementation
        score = calinski_harabasz_score(
            valid_embeddings,
            valid_labels
        )
        
        print(f"Calinski-Harabasz Score: {score:.4f}")
        return score
    
    def calculate_cluster_statistics(self):
        """
        Calculate detailed cluster statistics.
        
        Returns:
            dict: Statistics about clusters
        """
        print("\nCalculating Cluster Statistics...")
        
        unique_labels = set(self.labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(self.labels == -1)
        
        # Cluster sizes
        cluster_sizes = []
        for label in unique_labels:
            if label != -1:
                cluster_sizes.append(np.sum(self.labels == label))
        
        stats = {
            'n_clusters': n_clusters,
            'n_noise': int(n_noise),
            'noise_ratio': float(n_noise / len(self.labels)),
            'min_cluster_size': int(min(cluster_sizes)) if cluster_sizes else 0,
            'max_cluster_size': int(max(cluster_sizes)) if cluster_sizes else 0,
            'avg_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0,
            'median_cluster_size': float(np.median(cluster_sizes)) if cluster_sizes else 0,
            'std_cluster_size': float(np.std(cluster_sizes)) if cluster_sizes else 0,
        }
        
        print(f"  - Number of clusters: {stats['n_clusters']}")
        print(f"  - Noise ratio: {stats['noise_ratio']*100:.2f}%")
        print(f"  - Cluster size (min/avg/max): {stats['min_cluster_size']}/{stats['avg_cluster_size']:.1f}/{stats['max_cluster_size']}")
        
        return stats
    
    def evaluate_all_metrics(self):
        """
        Evaluate all clustering quality metrics.
        
        Returns:
            dict: All evaluation results
        """
        print("\n" + "="*80)
        print("OPTIMIZED CLUSTERING QUALITY EVALUATION")
        print("="*80)
        
        if self.embeddings is None:
            raise RuntimeError("Embeddings not loaded. Call generate_or_load_embeddings() first.")
        
        results = {}
        
        # 1. Silhouette Score
        results['silhouette_score'] = self.calculate_silhouette_score()
        
        # 2. Davies-Bouldin Index
        results['davies_bouldin_index'] = self.calculate_davies_bouldin_index()
        
        # 3. Calinski-Harabasz Score
        results['calinski_harabasz_score'] = self.calculate_calinski_harabasz_score()
        
        # 4. Cluster Statistics
        results['cluster_statistics'] = self.calculate_cluster_statistics()
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        
        # Overall Assessment
        print("\n" + "="*80)
        print("OVERALL ASSESSMENT")
        print("="*80)
        
        scores = []
        if results['silhouette_score'] is not None:
            scores.append(results['silhouette_score'])
        if results['davies_bouldin_index'] is not None:
            # Invert DBI (lower is better → higher score)
            scores.append(max(0, 1 - results['davies_bouldin_index']/2))
        
        if scores:
            avg_score = np.mean(scores)        
        stats = results['cluster_statistics']

        print("\nEvaluation Summary:")
        if results['silhouette_score'] is not None:
            sil = results['silhouette_score']
        print(f"  • Silhouette Score: {sil:.4f}")

        if results['davies_bouldin_index'] is not None:
            dbi = results['davies_bouldin_index']
        print(f"  • Davies-Bouldin Index: {dbi:.4f}")
        
        if results['calinski_harabasz_score'] is not None:
            ch = results['calinski_harabasz_score']
        print(f"  • Calinski-Harabasz Score: {ch:.2f}")

        print(f"\nData Distribution:")
        print(f"  • Total clusters: {stats['n_clusters']}")
        print(f"  • Noise ratio: {stats['noise_ratio']*100:.1f}%")
        print(f"  • Cluster size variation: {stats['std_cluster_size']:.1f} (std dev)")

def main():
    parser = argparse.ArgumentParser(
        description="Optimized clustering quality evaluation using sklearn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With pre-saved embeddings (FASTEST)
  python clustering_quality_evaluation_optimized.py clusters.csv --embeddings embeddings.npy
  
  # Generate embeddings on-the-fly
  python clustering_quality_evaluation_optimized.py clusters.csv --generate-embeddings
  
  # Save embeddings for future use
  python clustering_quality_evaluation_optimized.py clusters.csv --generate-embeddings --save-embeddings embeddings.npy
  
  # Save results to JSON
  python clustering_quality_evaluation_optimized.py clusters.csv --embeddings embeddings.npy --output results.json
        """
    )
    
    parser.add_argument("csv_file", help="Path to clustering results CSV file")
    parser.add_argument("--embeddings", "-e", help="Path to saved embeddings (.npy file)")
    parser.add_argument("--generate-embeddings", "-g", action='store_true',
                       help="Generate embeddings from scratch")
    parser.add_argument("--save-embeddings", "-s", help="Save generated embeddings to file")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                       help="Batch size for embedding generation (default: 32)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: CSV file not found: '{args.csv_file}'")
        sys.exit(1)
    
    if not args.embeddings and not args.generate_embeddings:
        print("❌ Error: Must provide either --embeddings or --generate-embeddings")
        print("Run with --help for usage examples")
        sys.exit(1)
    
    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = OptimizedClusteringEvaluator(args.csv_file)
    
    # Load or generate embeddings
    try:
        if args.generate_embeddings:
            evaluator.generate_or_load_embeddings(
                embedding_file=args.save_embeddings,
                batch_size=args.batch_size
            )
        else:
            evaluator.generate_or_load_embeddings(embedding_file=args.embeddings)
    except Exception as e:
        print(f"❌ Error loading/generating embeddings: {e}")
        sys.exit(1)
    
    # Run evaluation
    try:
        results = evaluator.evaluate_all_metrics()
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save results if requested
    if args.output:
        try:
            # Convert numpy types to Python types for JSON serialization
            results_json = {
                'silhouette_score': float(results['silhouette_score']) if results['silhouette_score'] is not None else None,
                'davies_bouldin_index': float(results['davies_bouldin_index']) if results['davies_bouldin_index'] is not None else None,
                'calinski_harabasz_score': float(results['calinski_harabasz_score']) if results['calinski_harabasz_score'] is not None else None,
                'cluster_statistics': results['cluster_statistics']
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Results saved to: {args.output}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save results: {e}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()