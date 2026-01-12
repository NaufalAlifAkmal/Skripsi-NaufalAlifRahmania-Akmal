import os
import re
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse, unquote, parse_qs
from transformers import BertTokenizer, BertModel
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# GPU optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT model globally
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
MODEL = BertModel.from_pretrained("bert-base-uncased").to(device)
MODEL.eval()

# ============================================================================
# URL PREPROCESSING
# ============================================================================

def extract_unique_urls_with_stats(url_list):
    """
    Extract URLs with frequency statistics for better analysis.
    """
    url_counts = Counter(url_list)
    unique_urls = []
    url_frequencies = []
    url_originals = {}
    
    for url in url_list:
        masked = re.sub(r'\d+', '<NUM>', url)
        
        if masked not in url_originals:
            unique_urls.append(masked)
            url_frequencies.append(url_counts[url])
            url_originals[masked] = [url]
        else:
            idx = unique_urls.index(masked)
            url_frequencies[idx] += url_counts[url]
            url_originals[masked].append(url)
    
    return unique_urls, url_frequencies, url_originals

def categorize_url(url):
    """
    Categorize URL into predefined types for cybersecurity analysis.
    """
    url_lower = url.lower()
    parsed = urlparse(url)
    path = parsed.path
    
    categories = {
        'static_resource': any(ext in path for ext in ['.css', '.js', '.jpg', '.png', '.gif', '.ico', '.woff', '.woff2', '.svg', '.ttf']),
        'api_endpoint': any(keyword in path for keyword in ['/api/', '/ajax', '/json', '/xml']),
        'filter_search': any(keyword in path for keyword in ['/filter', '/search', '/browse']),
        'product_page': '/product/' in path or '/item/' in path,
        'user_action': any(keyword in path for keyword in ['/login', '/logout', '/register', '/auth', '/signin']),
        'admin_panel': any(keyword in path for keyword in ['/admin', '/dashboard', '/panel', '/manage']),
        'blog_content': '/blog/' in path or '/article/' in path or '/post/' in path,
        'image_resource': '/image/' in path or '/img/' in path or '/media/' in path,
        'mobile_version': path.startswith('/m/') or '/mobile/' in path,
        'site_utility': any(keyword in path for keyword in ['/robots.txt', '/sitemap', '/favicon', '/manifest']),
        
        # SECURITY-SPECIFIC CATEGORIES
        'suspected_sqli': any(keyword in url_lower for keyword in ['union', 'select', 'drop', 'insert', 'update', 'delete', '--', '/*', '*/']),
        'suspected_xss': any(keyword in url_lower for keyword in ['<script', 'javascript:', 'onerror=', 'onload=']),
        'suspected_lfi': any(keyword in url_lower for keyword in ['../', '..\\', '/etc/passwd', '/proc/', 'file://']),
        'suspected_rfi': 'http://' in parsed.query or 'https://' in parsed.query or 'ftp://' in parsed.query,
        'suspected_traversal': '..' in path or '%2e%2e' in url_lower,
        'suspected_cmd_injection': any(keyword in url_lower for keyword in ['|', ';', '`', '$(']),
        'base64_encoded': bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', url)),
        'hex_encoded': bool(re.search(r'(%[0-9a-fA-F]{2}){4,}', url)),
    }
    
    return categories

def extract_url_structural_features(url):
    """
    Extract detailed structural features from URL.
    """
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    query = parsed.query
    
    query_params = parse_qs(query) if query else {}
    path_segments = [p for p in path.split('/') if p]
    path_depth = len(path_segments)
    
    char_diversity = len(set(url)) / max(len(url), 1)
    special_char_ratio = sum(1 for c in url if not c.isalnum() and c not in '/:?&=.-_') / max(len(url), 1)
    
    def calculate_entropy(s):
        if not s:
            return 0
        prob = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * np.log2(p) for p in prob if p > 0)
    
    features = {
        'path_depth': min(path_depth, 10),
        'path_length': min(len(path), 200),
        'url_length': min(len(url), 500),
        'query_length': min(len(query), 200),
        'num_query_params': min(len(query_params), 20),
        'char_diversity': char_diversity,
        'special_char_ratio': special_char_ratio,
        'entropy': calculate_entropy(url),
        'uppercase_ratio': sum(1 for c in url if c.isupper()) / max(len(url), 1),
        'digit_ratio': sum(1 for c in url if c.isdigit()) / max(len(url), 1),
        'has_query': 1 if query else 0,
        'has_fragment': 1 if parsed.fragment else 0,
        'has_extension': 1 if re.search(r'\.[a-z]{2,4}$', path) else 0,
        'num_slashes': url.count('/'),
        'num_dots': url.count('.'),
        'num_dashes': url.count('-'),
        'num_underscores': url.count('_'),
        'num_encoded_chars': len(re.findall(r'%[0-9a-fA-F]{2}', url)),
    }
    
    categories = categorize_url(url)
    features.update({f'cat_{k}': (1 if v else 0) for k, v in categories.items()})
    
    return features

def advanced_url_tokenization(url):
    """
    Advanced tokenization that preserves semantic meaning.
    """
    parsed = urlparse(url)
    path = unquote(parsed.path)
    query = unquote(parsed.query)
    
    tokens = []
    
    categories = categorize_url(url)
    for cat, present in categories.items():
        if present:
            tokens.append(f"TYPE_{cat}")
    
    path_segments = [p for p in path.strip('/').split('/') if p]
    tokens.append(f"DEPTH_{min(len(path_segments), 10)}")
    
    for i, segment in enumerate(path_segments):
        tokens.append(f"PATH{i}_{segment[:50]}")
        delimiters = r'[\-\_\.\+]'
        sub_tokens = re.split(delimiters, segment)
        tokens.extend([t for t in sub_tokens if t and len(t) > 1])
    
    if query:
        tokens.append("HAS_QUERY")
        query_params = parse_qs(query)
        for key, values in query_params.items():
            tokens.append(f"PARAM_{key}")
            for val in values:
                if val:
                    if val.isdigit():
                        tokens.append("VAL_NUMERIC")
                    elif len(val) > 20:
                        tokens.append("VAL_LONG")
                    else:
                        tokens.append(f"VAL_{val[:20]}")
    
    return tokens

# ============================================================================
# EMBEDDING GENERATION - HYBRID APPROACH
# ============================================================================

def generate_embeddings(url_list, batch_size=32, use_pca=True):
    """
    Generate hybrid embeddings combining BERT, structural, and statistical features.
    """
    print(f"Generating embeddings for {len(url_list)} URLs...")
    
    # 1. BERT Embeddings
    tokenized_urls = [" ".join(advanced_url_tokenization(url)) for url in url_list]
    bert_embeddings = []
    
    for i in range(0, len(tokenized_urls), batch_size):
        batch = tokenized_urls[i:i+batch_size]
        inputs = TOKENIZER(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = MODEL(**inputs)
        
        cls_emb = outputs.last_hidden_state[:, 0, :]
        mean_emb = outputs.last_hidden_state.mean(dim=1)
        combined = (cls_emb * 0.6 + mean_emb * 0.4)
        
        bert_embeddings.append(combined.cpu())
        
        if (i // batch_size) % 10 == 0:
            print(f"   Processed {min(i+batch_size, len(tokenized_urls))}/{len(tokenized_urls)} URLs")
    
    bert_embeddings = torch.cat(bert_embeddings, dim=0).numpy()
    bert_embeddings = normalize(bert_embeddings)
    
    # 2. Structural Features
    print("\nExtracting structural features...")
    structural_features = np.array([
        list(extract_url_structural_features(url).values())
        for url in url_list
    ])
    
    scaler = StandardScaler()
    structural_features = scaler.fit_transform(structural_features)
    
    # 3. Combine embeddings
    print("Combining embeddings...")
    combined_embeddings = np.hstack([
        bert_embeddings * 0.4,
        structural_features * 0.4,
        bert_embeddings[:, :50] * 0.2
    ])
    
    combined_embeddings = normalize(combined_embeddings)
    
    # 4. Apply PCA
    if use_pca and combined_embeddings.shape[1] > 100:
        print("Applying PCA for dimensionality reduction...")
        n_components = min(100, combined_embeddings.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=42)
        combined_embeddings = pca.fit_transform(combined_embeddings)
        combined_embeddings = normalize(combined_embeddings)
        print(f"   Reduced to {n_components} dimensions (explained variance: {pca.explained_variance_ratio_.sum():.2%})")
    
    return combined_embeddings

# ============================================================================
# OPTIMIZED HDBSCAN CLUSTERING
# ============================================================================

def optimize_clustering_params(n_samples, embeddings):
    """
    Auto-tune HDBSCAN parameters based on data size.
    """
    from sklearn.neighbors import NearestNeighbors
    sample_size = min(1000, n_samples)
    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]
    
    k = min(20, n_samples - 1)
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(sample_embeddings)
    distances, _ = nbrs.kneighbors(sample_embeddings)
    avg_distance = distances[:, 1:].mean()
    
    print(f"Data characteristics:")
    print(f"   Average k-NN distance: {avg_distance:.4f}")
    
    if n_samples < 1000:
        min_cluster_size = 5
        min_samples = 3
    elif n_samples < 5000:
        min_cluster_size = 10
        min_samples = 5
    elif n_samples < 15000:
        min_cluster_size = 15
        min_samples = 8
    else:
        min_cluster_size = 20
        min_samples = 10
    
    if avg_distance > 0.5:
        min_cluster_size = int(min_cluster_size * 0.7)
        min_samples = int(min_samples * 0.7)
    
    print(f"Optimized parameters:")
    print(f"   min_cluster_size: {min_cluster_size}")
    print(f"   min_samples: {min_samples}")
    
    return min_cluster_size, min_samples

def post_process_noise_reduction(embeddings, labels, threshold=0.35):
    """
    Reduce noise by reassigning points close to existing clusters.
    """
    noise_mask = labels == -1
    n_noise_before = noise_mask.sum()
    
    if n_noise_before == 0:
        return labels
    
    print(f"Post-processing {n_noise_before} noise points...")
    
    valid_clusters = [l for l in set(labels) if l != -1]
    
    if len(valid_clusters) == 0:
        return labels
    
    cluster_centroids = []
    for cluster_id in valid_clusters:
        mask = labels == cluster_id
        centroid = embeddings[mask].mean(axis=0)
        cluster_centroids.append(centroid)
    
    cluster_centroids = np.array(cluster_centroids)
    
    noise_indices = np.where(noise_mask)[0]
    reassigned = 0
    
    for idx in noise_indices:
        point = embeddings[idx].reshape(1, -1)
        distances = 1 - np.dot(cluster_centroids, point.T).flatten()
        min_dist = distances.min()
        
        if min_dist < threshold:
            nearest_cluster = valid_clusters[distances.argmin()]
            labels[idx] = nearest_cluster
            reassigned += 1
    
    n_noise_after = (labels == -1).sum()
    print(f"Reassigned {reassigned} points")
    print(f"   Noise: {n_noise_before} -> {n_noise_after} ({n_noise_after/len(labels)*100:.1f}%)")
    
    return labels

# ============================================================================
# MAIN CLUSTERING FUNCTION
# ============================================================================

def cluster_urls_from_log(url_list, out_path, min_cluster_size=None, min_samples=None, 
                          auto_tune=True, use_pca=True, save_embeddings=True):
    """
    Main clustering function optimized for cybersecurity investigation.
    """
    embedding_file = None
    print("\n" + "="*80)
    print("CYBERSECURITY LOG CLUSTERING SYSTEM")
    print("="*80 + "\n")
    
    # Step 1: Extract URLs
    print("Step 1: Extracting unique URL patterns...")
    unique_urls, frequencies, url_originals = extract_unique_urls_with_stats(url_list)
    print(f"Found {len(unique_urls)} unique URL patterns from {len(url_list)} requests")
    print(f"   Total unique requests: {sum(frequencies)}")
    print(f"   Average requests per pattern: {sum(frequencies)/len(unique_urls):.2f}")
    
    # Step 2: Generate embeddings
    print(f"\nStep 2: Generating hybrid embeddings...")
    embeddings = generate_embeddings(unique_urls, use_pca=use_pca)
    print(f"Generated embeddings: shape {embeddings.shape}")
    
    if save_embeddings:
        embedding_file = out_path.replace('.csv', '_embeddings.npy')
        np.save(embedding_file, embeddings)
        print(f"✅ Embeddings saved to: {embedding_file}")

    # Step 3: Optimize parameters
    print(f"\nStep 3: Optimizing clustering parameters...")
    if auto_tune or min_cluster_size is None or min_samples is None:
        min_cluster_size, min_samples = optimize_clustering_params(len(unique_urls), embeddings)
    
    # Step 4: Perform clustering
    print(f"\nStep 4: Performing HDBSCAN clustering...")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='cosine',
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.0,
        alpha=1.0,
        algorithm='auto',
        leaf_size=40,
        n_jobs=-1
    )
    
    labels = clusterer.fit_predict(embeddings)
    print(f"Initial clustering complete")
    print(f"   Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"   Noise points: {(labels == -1).sum()} ({(labels == -1).sum()/len(labels)*100:.1f}%)")
    
    # Step 5: Post-process noise
    print(f"\nStep 5: Post-processing noise reduction...")
    labels = post_process_noise_reduction(embeddings, labels, threshold=0.35)
    
    # Step 6: Organize results
    print(f"\nStep 6: Organizing results...")
    unique_labels = set(labels)
    clustered_urls = {label: [] for label in unique_labels}
    
    for idx, label in enumerate(labels):
        clustered_urls[label].append(unique_urls[idx])
    
    # Step 7: Generate output files
    print(f"\nStep 7: Generating output files...")
    
    # Output 1: CSV file (format: masked,cluster)
    df_results = pd.DataFrame({
        'masked': unique_urls,
        'cluster': labels
    }).sort_values(by='cluster')
    
    df_results.to_csv(out_path, index=False, encoding='utf-8')
    
    # Output 2: TXT file (format seperti GitHub)
    with open(f"{out_path}.txt", "w", encoding="utf-8") as f:
        # Noise points first
        if -1 in clustered_urls:
            f.write(f"Noise Points (Outliers):\n")
            for url in sorted(clustered_urls[-1]):
                f.write(f"  {url}\n")
            f.write("\n")
        
        # Clusters sorted by ID
        sorted_clusters = sorted(
            [(k, v) for k, v in clustered_urls.items() if k != -1],
            key=lambda x: x[0]
        )
        
        for cluster_id, urls in sorted_clusters:
            f.write(f"Cluster {cluster_id}:\n")
            for url in sorted(urls):
                f.write(f"  {url}\n")
            f.write("\n")
        
        # Summary
        n_clusters = len([l for l in unique_labels if l != -1])
        n_noise = len(clustered_urls.get(-1, []))
        
        f.write(f"--- Clustering Summary ---\n")
        f.write(f"Total clusters found: {n_clusters}\n")
        f.write(f"Noise points (outliers): {n_noise}\n")
        f.write(f"Total URLs processed: {len(unique_urls)}\n")
    
    # Final summary
    n_clusters = len([l for l in unique_labels if l != -1])
    n_noise = (labels == -1).sum()
    
    print("\n" + "="*80)
    print("✅ CLUSTERING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results:")
    print(f"   Total clusters: {n_clusters}")
    print(f"   Clustered patterns: {len(unique_urls) - n_noise}/{len(unique_urls)} ({(len(unique_urls)-n_noise)/len(unique_urls)*100:.1f}%)")
    print(f"   Noise patterns: {n_noise} ({n_noise/len(unique_urls)*100:.1f}%)")
    print(f"\nOutput files:")
    print(f"   CSV results: {out_path}")
    print(f"   Text details: {out_path}.txt")
    if save_embeddings and embedding_file:
        print(f"   Embeddings: {embedding_file}")
    print("="*80 + "\n")

# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cybersecurity Log Clustering System for Attack Investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-optimized clustering (RECOMMENDED)
  python hdbscan_improved.py inputs/preprocessing_10jtlog.txt outputs/clusters.csv
  
  # Manual parameters
  python hdbscan_improved.py inputs/preprocessing_10jtlog.txt outputs/clusters.csv --min-cluster-size 20 --min-samples 10
  
  # For large datasets
  python hdbscan_improved.py inputs/preprocessing_10jtlog.txt outputs/clusters.csv --min-cluster-size 30 --min-samples 15
        """
    )
    
    parser.add_argument("in_file", help="Preprocessed log file to analyze (one URL per line)")
    parser.add_argument("out_file", help="Output CSV file path")
    parser.add_argument("--min-cluster-size", type=int, default=None,
                       help="Minimum cluster size (auto-tuned if not specified)")
    parser.add_argument("--min-samples", type=int, default=None,
                       help="Minimum samples per cluster (auto-tuned if not specified)")
    parser.add_argument("--no-auto-tune", action='store_true',
                       help="Disable automatic parameter tuning")
    parser.add_argument("--no-pca", action='store_true',
                       help="Disable PCA dimensionality reduction")
    parser.add_argument("--noise-threshold", type=float, default=0.35,
                       help="Threshold for noise reassignment (0.2-0.5, default: 0.35)")
    parser.add_argument("--save-embeddings", action='store_true', default=True,
                       help="Save embeddings to .npy file (default: True)")
    parser.add_argument("--no-save-embeddings", action='store_true',
                       help="Don't save embeddings")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.in_file):
        print(f"❌ Error: Input file not found: '{args.in_file}'")
        sys.exit(1)
    
    out_dir = os.path.dirname(args.out_file)
    if out_dir and not os.path.exists(out_dir):
        print(f"❌ Error: Output directory not found: '{out_dir}'")
        sys.exit(1)
    
    # Load log data
    print("Loading preprocessed log file...")
    try:
        with open(args.in_file, 'r', encoding='utf-8') as f:
            url_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(url_list)} URL entries from {args.in_file}")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Run clustering
    try:
        cluster_urls_from_log(
            url_list=url_list,
            out_path=args.out_file,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            auto_tune=not args.no_auto_tune,
            use_pca=not args.no_pca,
            save_embeddings=not args.no_save_embeddings
        )
    except Exception as e:
        print(f"\nError during clustering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)