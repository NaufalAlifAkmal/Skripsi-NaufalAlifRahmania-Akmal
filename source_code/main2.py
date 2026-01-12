import os
import re
import sys
import argparse
import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse, unquote, parse_qs
from decoder import parse_dec_file_to_dataframe, decoding_stats
from bot import stats as bot_stats

# URL PROCESSING FUNCTIONS
def extract_unique_urls_with_stats(df):
    """
    Extract URLs with frequency statistics for better analysis.
    """
    url_counts = df['url'].value_counts()
    unique_urls = []
    url_frequencies = []
    url_originals = {}
    
    # Count URLs after normalization
    normalized_urls = []
    
    for url in df['url'].unique():
        masked = re.sub(r'\d+', '<NUM>', url)
        normalized_urls.append(masked)
        
        if masked not in url_originals:
            unique_urls.append(masked)
            url_frequencies.append(url_counts[url])
            url_originals[masked] = [url]
        else:
            idx = unique_urls.index(masked)
            url_frequencies[idx] += url_counts[url]
            url_originals[masked].append(url)
    
    # Update statistics
    decoding_stats['urls_after_normalization'] = len(normalized_urls)
    decoding_stats['unique_urls_final'] = len(unique_urls)
    
    return unique_urls, url_frequencies, url_originals

# SIMPLIFIED URL OUTPUT FUNCTION (NO CLUSTERING)
def output_urls_only(df, out_path):
    """
    Simplified function to output only URLs without clustering.
    """
    print("\n" + "="*80)
    print("URL EXTRACTION SYSTEM")
    print("="*80 + "\n")
    
    # Extract URLs
    print("Extracting URLs...")
    unique_urls, frequencies, url_originals = extract_unique_urls_with_stats(df)
    print(f"Found {len(unique_urls)} unique URL patterns from {len(df)} requests")
    
    # Calculate reduction percentage
    total_requests = bot_stats['total_requests']
    urls_after_bot_filtering = decoding_stats['urls_after_bot_filtering']
    reduction_percentage = ((total_requests - urls_after_bot_filtering) / total_requests * 100) if total_requests > 0 else 0
    
    # Display statistics
    print("\n" + "-"*80)
    print("STATISTIK PEMROSESAN")
    print("-"*80)
    print(f"Jumlah URL valid setelah parsing          : {decoding_stats['urls_after_parsing']}")
    print(f"Jumlah URL setelah filtering bot          : {decoding_stats['urls_after_bot_filtering']}")
    print(f"Jumlah URL setelah normalisasi <NUM>      : {decoding_stats['urls_after_normalization']}")
    print(f"Jumlah URL unik akhir                     : {decoding_stats['unique_urls_final']}")
    print(f"Persentase reduksi data                   : {reduction_percentage:.2f}%")
    
    # Output: Simple text file with URLs only
    print(f"\nGenerating output file with URLs only...")
    with open(out_path, "w", encoding="utf-8") as f:
        for url in sorted(unique_urls):
            f.write(f"{url}\n")
    
    print("\n" + "="*80)
    print("✅ URL EXTRACTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results:")
    print(f"   Unique URLs extracted: {len(unique_urls)}")
    print(f"\nOutput file:")
    print(f"   URLs only: {out_path}")
    print("="*80 + "\n")

# CLI INTERFACE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple URL Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("in_file", help="NGINX log file to analyze")
    parser.add_argument("out_file", help="Output file path for URLs")
    
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
    print("Loading log file...")
    try:
        df = parse_dec_file_to_dataframe(args.in_file)
        print(f"Loaded {len(df)} log entries from {args.in_file}")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Run URL extraction only
    try:
        output_urls_only(
            df=df,
            out_path=args.out_file
        )
    except Exception as e:
        print(f"\nError during URL extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)