"""
Complete Pipeline Orchestrator for Web Log Clustering
FIXED VERSION - Optimized for Large Datasets
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json
import html  # IMPORTANT: Use html.escape() instead of manual replace

class PipelineOrchestrator:
    def __init__(self, log_file, output_dir, chunk_size=20000):
        self.log_file = log_file
        self.output_dir = Path(output_dir)
        self.chunk_size = int(chunk_size)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / "temp_chunks"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.preprocessed_file = self.output_dir / "urls_extracted.txt"
        self.original_urls_file = self.output_dir / "urls_original.txt"
        self.final_csv = self.output_dir / "clusters_final.csv"
        self.final_txt_masked = self.output_dir / "clusters_masked.txt"
        self.final_txt_original = self.output_dir / "clusters_original.txt"
        
        self.start_time = None
        self.step_times = {}
        
    def print_header(self, title):
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80 + "\n")
    
    def print_step(self, step_num, title):
        print(f"\n{'─'*80}")
        print(f"STEP {step_num}: {title}")
        print(f"{'─'*80}\n")
    
    def run_command(self, cmd, step_name):
        print(f"Executing: {' '.join(cmd)}\n")
        step_start = time.time()
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            step_time = time.time() - step_start
            self.step_times[step_name] = step_time
            print(f"\n✅ {step_name} completed in {step_time:.2f}s\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error in {step_name}")
            print(f"Command failed with exit code {e.returncode}")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error in {step_name}: {e}")
            return False
    
    def step1_preprocessing(self):
        self.print_step(1, "URL EXTRACTION FROM LOG FILE")
        
        cmd = [sys.executable, "main2.py", str(self.log_file), str(self.preprocessed_file)]
        success = self.run_command(cmd, "URL Extraction")
        
        if success:
            print("\nSaving original URLs for investigation...")
            from decoder import parse_dec_file_to_dataframe
            df = parse_dec_file_to_dataframe(str(self.log_file))
            
            with open(self.original_urls_file, 'w', encoding='utf-8') as f:
                for url in df['url']:
                    f.write(f"{url}\n")
            
            print(f"✅ Saved {len(df)} original URLs to: {self.original_urls_file}")
        
        return success
    
    def step2_chunking(self):
        self.print_step(2, "SPLITTING INTO CHUNKS")
        
        with open(self.preprocessed_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        total_urls = len(urls)
        n_chunks = (total_urls + self.chunk_size - 1) // self.chunk_size
        
        print(f"Total URLs: {total_urls}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Number of chunks: {n_chunks}\n")
        
        chunk_files = []
        for i in range(n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, total_urls)
            chunk_urls = urls[start_idx:end_idx]
            
            chunk_file = self.temp_dir / f"chunk_{i:03d}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(chunk_urls))
            
            chunk_files.append(chunk_file)
            print(f"Created {chunk_file.name}: {len(chunk_urls)} URLs")
        
        self.chunk_files = chunk_files
        print(f"\n✅ Chunking completed: {len(chunk_files)} chunks created")
        return True
    
    def step3_clustering_chunks(self):
        self.print_step(3, "CLUSTERING EACH CHUNK")
        
        print(f"Processing {len(self.chunk_files)} chunks...\n")
        
        for i, chunk_file in enumerate(self.chunk_files):
            output_csv = self.temp_dir / f"cluster_chunk_{i:03d}.csv"
            print(f"[{i+1}/{len(self.chunk_files)}] Clustering {chunk_file.name}...\n")
            
            cmd = [sys.executable, "hdbscan_improved.py", str(chunk_file), str(output_csv)]
            
            if not self.run_command(cmd, f"Chunk {i+1} Clustering"):
                return False
        
        print(f"\n✅ All chunks clustered successfully")
        return True
    
    def step4_merge_results(self):
        self.print_step(4, "MERGING CLUSTERING RESULTS")
        
        cluster_files = sorted(self.temp_dir.glob("cluster_chunk_*.csv"))
        
        if not cluster_files:
            print("❌ No cluster files found!")
            return False
        
        print(f"Found {len(cluster_files)} cluster files to merge\n")
        
        import pandas as pd
        
        all_results = []
        global_cluster_id = 0
        
        for i, cluster_file in enumerate(cluster_files):
            df = pd.read_csv(cluster_file)
            print(f"[{i+1}/{len(cluster_files)}] {cluster_file.name}: {len(df)} URLs")
            
            df['cluster'] = df['cluster'].apply(
                lambda x: -1 if x == -1 else x + global_cluster_id
            )
            
            all_results.append(df)
            
            max_cluster = df[df['cluster'] != -1]['cluster'].max()
            if pd.notna(max_cluster):
                global_cluster_id = int(max_cluster) + 1
        
        merged_df = pd.concat(all_results, ignore_index=True)
        merged_df = merged_df.sort_values(by='cluster').reset_index(drop=True)
        
        print(f"\nLoading original URLs for investigation...")
        with open(self.original_urls_file, 'r', encoding='utf-8') as f:
            all_original_urls = [line.strip() for line in f if line.strip()]
        
        import re
        url_mapping = {}
        for original_url in all_original_urls:
            masked = re.sub(r'\d+', '<NUM>', original_url)
            if masked not in url_mapping:
                url_mapping[masked] = []
            url_mapping[masked].append(original_url)
        
        for key in url_mapping:
            url_mapping[key] = list(set(url_mapping[key]))
        
        merged_df['original_examples'] = merged_df['masked'].apply(
            lambda x: '|||'.join(url_mapping.get(x, [x])[:5])
        )
        merged_df['original_count'] = merged_df['masked'].apply(
            lambda x: len(url_mapping.get(x, [x]))
        )
        
        print(f"✅ Loaded {len(all_original_urls):,} original URLs")
        print(f"✅ Created mapping for {len(url_mapping):,} unique patterns")
        
        merged_df.to_csv(self.final_csv, index=False, encoding='utf-8')
        print(f"\n✅ Merged CSV saved: {self.final_csv}")
        
        mapping_file = self.output_dir / "url_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(url_mapping, f, indent=2, ensure_ascii=False)
        print(f"✅ URL mapping saved: {mapping_file}")
        
        # Generate text summaries
        clusters = sorted([c for c in merged_df['cluster'].unique() if c != -1])
        noise_df = merged_df[merged_df['cluster'] == -1]
        
        # Masked version
        print(f"\nGenerating masked text summary...")
        with open(self.final_txt_masked, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CLUSTERING RESULTS - MASKED PATTERNS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total URL patterns: {len(merged_df)}\n")
            f.write(f"Total clusters: {len(clusters)}\n")
            f.write(f"Noise patterns: {len(noise_df)}\n\n")
            
            if len(noise_df) > 0:
                f.write(f"Noise Points (Outliers): {len(noise_df)} patterns\n")
                f.write("-" * 80 + "\n")
                for url in sorted(noise_df['masked'].tolist()):
                    f.write(f"  {url}\n")
                f.write("\n")
            
            for cluster_id in clusters:
                cluster_urls = merged_df[merged_df['cluster'] == cluster_id]['masked'].tolist()
                f.write(f"Cluster {int(cluster_id)}: {len(cluster_urls)} patterns\n")
                for url in sorted(cluster_urls):
                    f.write(f"  {url}\n")
                f.write("\n")
        
        print(f"✅ Masked text summary: {self.final_txt_masked}\n")
        
        # Original version
        print(f"Generating original text summary...")
        with open(self.final_txt_original, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CLUSTERING RESULTS - ORIGINAL URLs\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total clusters: {len(clusters)}\n")
            f.write(f"Total requests: {len(all_original_urls):,}\n\n")
            
            if len(noise_df) > 0:
                noise_originals = []
                for masked_pattern in noise_df['masked'].tolist():
                    noise_originals.extend(url_mapping.get(masked_pattern, [masked_pattern]))
                
                f.write(f"Noise Points (Outliers): {len(noise_originals)} URLs\n")
                f.write("-" * 80 + "\n")
                for url in sorted(set(noise_originals)):
                    f.write(f"  {url}\n")
                f.write("\n")
            
            for cluster_id in clusters:
                cluster_patterns = merged_df[merged_df['cluster'] == cluster_id]['masked'].tolist()
                cluster_originals = []
                for pattern in cluster_patterns:
                    cluster_originals.extend(url_mapping.get(pattern, [pattern]))
                
                seen = set()
                unique_originals = []
                for url in cluster_originals:
                    if url not in seen:
                        seen.add(url)
                        unique_originals.append(url)
                
                f.write(f"Cluster {int(cluster_id)}: {len(unique_originals)} URLs\n")
                for url in sorted(unique_originals):
                    f.write(f"  {url}\n")
                f.write("\n")
        
        print(f"✅ Original text summary: {self.final_txt_original}")
        
        n_clusters = len(clusters)
        n_noise = len(noise_df)
        n_clustered = len(merged_df) - n_noise
        
        print(f"\nFinal Statistics:")
        print(f"  Total URL patterns: {len(merged_df)}")
        print(f"  Total requests: {len(all_original_urls):,}")
        print(f"  Clusters: {n_clusters}")
        print(f"  Clustered patterns: {n_clustered} ({n_clustered/len(merged_df)*100:.1f}%)")
        print(f"  Noise patterns: {n_noise} ({n_noise/len(merged_df)*100:.1f}%)")
        
        return True
    
    def step5_generate_html_report(self):
        self.print_step(5, "GENERATING HTML REPORT WITH CHARTS")
        
        html_start_time = time.time()
        
        import pandas as pd
        print(f"Loading clustering results from CSV...")
        df = pd.read_csv(self.final_csv)
        
        print(f"Calculating statistics...")
        clusters = sorted([c for c in df['cluster'].unique() if c != -1])
        noise_urls = df[df['cluster'] == -1]
        n_clusters = len(clusters)
        n_noise = len(noise_urls)
        n_clustered = len(df) - n_noise
        
        print(f"\nData Summary:")
        print(f"  Total URLs: {len(df):,}")
        print(f"  Total clusters: {n_clusters}")
        print(f"  Noise URLs: {n_noise:,}")
        
        # OPTIMIZATION: Limit display untuk dataset besar
        MAX_CLUSTERS_TO_DISPLAY = 100  # Hanya tampilkan 100 cluster terbesar
        MAX_PATTERNS_PER_CLUSTER = 20  # Maksimal 20 pattern per cluster
        MAX_NOISE_TO_DISPLAY = 50      # Maksimal 50 noise patterns
        
        print(f"\n⚠️  OPTIMIZATION APPLIED:")
        print(f"  - Max clusters to display: {MAX_CLUSTERS_TO_DISPLAY}")
        print(f"  - Max patterns per cluster: {MAX_PATTERNS_PER_CLUSTER}")
        print(f"  - Max noise patterns: {MAX_NOISE_TO_DISPLAY}")
        
        # Prepare chart data
        print(f"\nPreparing chart data...")
        chart_data = self._prepare_chart_data(df, clusters)
        
        html_file = self.output_dir / "report.html"
        
        print(f"\nGenerating HTML report...")
        try:
            # IMPORTANT: Gunakan buffer yang lebih besar
            with open(html_file, 'w', encoding='utf-8', buffering=1024*1024) as f:  # 1MB buffer
                print(f"  [1/6] Writing HTML header...")
                f.write(self._get_html_header())
                f.flush()
                
                print(f"  [2/6] Writing summary section...")
                f.write(self._get_summary_section(df, clusters, n_noise, n_clustered))
                f.flush()
                
                print(f"  [3/6] Writing charts section...")
                f.write(self._get_charts_section(chart_data))
                f.flush()
                
                print(f"  [4/6] Writing clusters section...")
                self._write_clusters_section_streaming(
                    f, df, clusters, 
                    max_clusters=MAX_CLUSTERS_TO_DISPLAY,
                    max_patterns=MAX_PATTERNS_PER_CLUSTER
                )
                f.flush()
                
                print(f"  [5/6] Writing noise section...")
                self._write_noise_section_streaming(
                    f, df,
                    max_display=MAX_NOISE_TO_DISPLAY
                )
                f.flush()
                
                print(f"  [6/6] Writing HTML footer...")
                f.write(self._get_html_footer())
                f.flush()
            
            file_size = html_file.stat().st_size
            print(f"\n✅ HTML report generated successfully!")
            print(f"  File: {html_file}")
            print(f"  Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            
            # IMPORTANT: Warn if file is too large
            if file_size > 50 * 1024 * 1024:  # > 50MB
                print(f"\n⚠️  WARNING: HTML file is very large ({file_size/1024/1024:.1f} MB)")
                print(f"  Your browser might have difficulty rendering it.")
                print(f"  Consider using CSV/TXT files for detailed analysis.")
            
            html_time = time.time() - html_start_time
            self.step_times["HTML Report"] = html_time
            print(f"  Time: {html_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error generating HTML report: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_chart_data(self, df, clusters):
        """Prepare data for charts"""
        n_noise = len(df[df['cluster'] == -1])
        n_clustered = len(df) - n_noise
        
        cluster_sizes = []
        for c in clusters:
            cluster_data = df[df['cluster'] == c]
            pattern_count = len(cluster_data)
            request_count = int(cluster_data['original_count'].sum())
            cluster_sizes.append({
                'cluster_id': int(c),
                'pattern_count': int(pattern_count),
                'request_count': request_count
            })
        
        cluster_sizes.sort(key=lambda x: x['request_count'], reverse=True)
        top_10_clusters = cluster_sizes[:10]
        
        scatter_data = []
        for item in cluster_sizes:
            scatter_data.append({
                'x': int(item['pattern_count']),
                'y': int(item['request_count']),
                'cluster_id': int(item['cluster_id'])
            })
        
        return {
            'pie': {
                'clustered': int(n_clustered),
                'noise': int(n_noise)
            },
            'bar': top_10_clusters
        }
    
    def _get_html_header(self):
        """Generate HTML header with CSS styling"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>URL Clustering Report - Security Investigation</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }
        .stat-card:hover { transform: translateY(-5px); box-shadow: 0 6px 12px rgba(0,0,0,0.15); }
        .stat-number { font-size: 3em; font-weight: bold; color: #667eea; margin: 10px 0; }
        .stat-label { color: #666; font-size: 1.1em; text-transform: uppercase; letter-spacing: 1px; }
        .charts-section { padding: 40px; background: #f8f9fa; }
        .charts-title {
            font-size: 2em;
            color: #2c3e50;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            text-align: center;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        .chart-container { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .chart-title { font-size: 1.3em; color: #2c3e50; margin-bottom: 20px; text-align: center; font-weight: bold; }
        .chart-canvas { position: relative; height: 300px; }
        .content { padding: 40px; }
        .section { margin-bottom: 40px; }
        .section-title {
            font-size: 2em;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        .cluster-card {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            margin-bottom: 20px;
            overflow: hidden;
            transition: all 0.3s;
        }
        .cluster-card:hover { border-color: #667eea; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2); }
        .cluster-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .cluster-header:hover { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); }
        .cluster-title { font-size: 1.5em; font-weight: bold; }
        .cluster-count { background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 20px; font-size: 1.1em; }
        .cluster-body { padding: 20px; display: none; background: #f8f9fa; }
        .cluster-body.active { display: block; }
        .noise-section { background: #fff3cd; border: 2px solid #ffc107; border-radius: 10px; padding: 20px; }
        .noise-section .section-title { color: #856404; border-bottom-color: #ffc107; }
        .show-more-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            margin-top: 15px;
            transition: all 0.3s;
        }
        .show-more-btn:hover { background: #764ba2; transform: scale(1.05); }
        .footer { background: #2c3e50; color: white; text-align: center; padding: 20px; font-size: 0.9em; }
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }
        .badge-success { background: #28a745; color: white; }
        .badge-warning { background: #ffc107; color: #333; }
        .badge-danger { background: #dc3545; color: white; }
        .pattern-block {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            transition: all 0.2s;
        }
        .pattern-block:hover { border-color: #667eea; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15); }
        .pattern-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .url-pattern {
            background: #f8f9fa;
            padding: 4px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #667eea;
            word-break: break-all;
        }
        .occurrence-badge {
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            white-space: nowrap;
        }
        .original-examples {
            padding: 10px;
            background: #f8f9fa;
            border-left: 3px solid #667eea;
            border-radius: 4px;
            margin-top: 10px;
        }
        .original-examples strong { color: #2c3e50; font-size: 0.9em; display: block; margin-bottom: 8px; }
        .example-list { list-style: none; margin: 0; padding: 0; }
        .example-list li {
            padding: 6px 10px;
            margin-bottom: 4px;
            background: white;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            color: #495057;
            word-break: break-all;
            border-left: 2px solid #667eea;
        }
        .example-list li:hover { background: #e8f0fe; }
        .example-list li em { color: #6c757d; font-style: italic; }
        .noise-pattern { border-color: #ffc107; background: #fffbf0; }
        .noise-pattern .original-examples { border-left-color: #ffc107; }
        .noise-pattern .example-list li { border-left-color: #ffc107; }
        .warning-box {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            color: #856404;
        }
        .warning-box strong { color: #856404; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 URL Clustering Report</h1>
            <p>Security Investigation Dashboard</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
"""
    
    def _get_summary_section(self, df, clusters, n_noise, n_clustered):
        """Generate summary statistics section"""
        df_len = len(df)
        clustered_pct = (n_clustered / df_len * 100) if df_len > 0 else 0
        noise_pct = (n_noise / df_len * 100) if df_len > 0 else 0
        
        return f"""
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Total URLs</div>
                <div class="stat-number">{df_len:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Clusters Found</div>
                <div class="stat-number">{len(clusters)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Clustered URLs</div>
                <div class="stat-number">{n_clustered:,}</div>
                <span class="badge badge-success">{clustered_pct:.1f}%</span>
            </div>
            <div class="stat-card">
                <div class="stat-label">Noise Points</div>
                <div class="stat-number">{n_noise:,}</div>
                <span class="badge badge-warning">{noise_pct:.1f}%</span>
            </div>
        </div>
"""
    
    def _get_charts_section(self, chart_data):
        """Generate charts section with data embedded"""
        chart_data_json = json.dumps(chart_data)
        
        return f"""
        <div class="charts-section">
            <h2 class="charts-title">📊 Visual Analytics</h2>
            
            <div class="charts-grid">
                <div class="chart-container">
                    <div class="chart-title">Clustered vs Noise Distribution</div>
                    <div class="chart-canvas">
                        <canvas id="pieChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Top 10 Largest Clusters</div>
                    <div class="chart-canvas">
                        <canvas id="barChart"></canvas>
                    </div>
                </div>
                
            </div>
        </div>
        
        <script>
            const chartData = {chart_data_json};
        </script>
"""
    
    def _write_clusters_section_streaming(self, file_handle, df, clusters, max_clusters=100, max_patterns=20):
        """FIXED: Write clusters section with proper HTML escaping and limits"""
        
        # Calculate cluster sizes
        print(f"     Analyzing cluster sizes...")
        cluster_sizes = []
        for c in clusters:
            cluster_data = df[df['cluster'] == c]
            pattern_count = len(cluster_data)
            request_count = int(cluster_data['original_count'].sum())
            cluster_sizes.append((int(c), pattern_count, request_count))
        
        cluster_sizes.sort(key=lambda x: x[2], reverse=True)
        
        # Apply limit
        total_clusters = len(cluster_sizes)
        cluster_sizes = cluster_sizes[:max_clusters]
        
        file_handle.write("""
        <div class="content">
            <div class="section">
                <h2 class="section-title">📊 Discovered Clusters</h2>
                <p style="margin-bottom: 20px; color: #666;">
                    Click on any cluster to expand and view URL patterns with original examples
                </p>
""")
        
        # Show warning if truncated
        if total_clusters > max_clusters:
            file_handle.write(f"""
                <div class="warning-box">
                    <strong>⚠️ Display Limit Applied</strong><br>
                    Showing top {max_clusters} of {total_clusters} clusters (sorted by request count).<br>
                    For complete data, please refer to: <code>clusters_final.csv</code> and <code>clusters_masked.txt</code>
                </div>
""")
        
        # Write each cluster
        for idx, (cluster_id, pattern_count, request_count) in enumerate(cluster_sizes):
            if (idx + 1) % 10 == 0:
                print(f"     Processing cluster {idx+1}/{len(cluster_sizes)}...")
            
            cluster_data = df[df['cluster'] == cluster_id]
            
            # Determine priority
            if request_count > 1000:
                badge_class = "badge-danger"
                priority = "🔴 HIGH PRIORITY"
            elif request_count > 100:
                badge_class = "badge-warning"
                priority = "🟡 MEDIUM"
            else:
                badge_class = "badge-success"
                priority = "🟢 LOW"
            
            # Write cluster header
            file_handle.write(f"""
                <div class="cluster-card">
                    <div class="cluster-header" onclick="toggleCluster({cluster_id})">
                        <div class="cluster-title">
                            Cluster #{cluster_id} - {priority}
                            <span class="badge {badge_class}">{pattern_count} patterns | {request_count:,} requests</span>
                        </div>
                        <div class="cluster-count">▼</div>
                    </div>
                    <div class="cluster-body" id="cluster-{cluster_id}">
""")
            
            # Apply pattern limit
            display_limit = min(max_patterns, len(cluster_data))
            
            for _, row in cluster_data.head(display_limit).iterrows():
                masked_url = str(row['masked'])
                original_count = int(row['original_count'])
                
                # Get original examples
                original_examples_str = row.get('original_examples', masked_url)
                if original_examples_str and '|||' in str(original_examples_str):
                    original_examples = str(original_examples_str).split('|||')[:3]
                else:
                    original_examples = [str(original_examples_str)] if original_examples_str else [masked_url]
                
                # CRITICAL FIX: Use html.escape() instead of manual replace
                url_escaped = html.escape(masked_url, quote=True)
                
                file_handle.write(f"""
                        <div class="pattern-block">
                            <div class="pattern-header">
                                <div>
                                    <strong>Pattern:</strong> <code class="url-pattern">{url_escaped}</code>
                                </div>
                                <span class="occurrence-badge">{original_count:,} occurrences</span>
                            </div>
                            <div class="original-examples">
                                <strong>🔍 Original URLs (for investigation):</strong>
                                <ul class="example-list">
""")
                
                # Show original examples
                for ex in original_examples:
                    if ex and str(ex).strip():
                        ex_escaped = html.escape(str(ex), quote=True)
                        file_handle.write(f'                                    <li>{ex_escaped}</li>\n')
                
                if original_count > len(original_examples):
                    file_handle.write(f'                                    <li><em>... and {original_count - len(original_examples)} more occurrences</em></li>\n')
                
                file_handle.write("""
                                </ul>
                            </div>
                        </div>
""")
            
            # Show more button if needed
            if len(cluster_data) > display_limit:
                remaining = len(cluster_data) - display_limit
                file_handle.write(f"""
                        <div style="text-align: center; padding: 15px;">
                            <button class="show-more-btn" onclick="alert('Showing {display_limit} of {len(cluster_data)} patterns\\n\\nFull data available in:\\n- clusters_final.csv\\n- clusters_masked.txt')">
                                📄 {remaining} more patterns (view in CSV/TXT)
                            </button>
                        </div>
""")
            
            file_handle.write("""
                    </div>
                </div>
""")
            
            # Flush every 5 clusters
            if (idx + 1) % 5 == 0:
                file_handle.flush()
        
        file_handle.write("""
            </div>
""")
        print(f"     ✅ {len(cluster_sizes)} clusters written (of {total_clusters} total)")
    
    def _write_noise_section_streaming(self, file_handle, df, max_display=50):
        """FIXED: Write noise section with proper HTML escaping and limits"""
        noise_df = df[df['cluster'] == -1]
        
        if len(noise_df) == 0:
            print(f"     No noise points found")
            file_handle.write("        </div>\n")
            return
        
        print(f"     Writing noise section...")
        
        total_noise = len(noise_df)
        total_noise_requests = int(noise_df['original_count'].sum())
        
        file_handle.write(f"""
            <div class="section">
                <div class="noise-section">
                    <h2 class="section-title">⚠️ Noise Points (Outliers)</h2>
                    <p style="margin-bottom: 20px; color: #856404;">
                        These URL patterns don't fit into any cluster. Total: <strong>{total_noise} patterns</strong> | <strong>{total_noise_requests:,} requests</strong>
                        <br>They might be unique attacks, reconnaissance attempts, or false positives.
                    </p>
""")
        
        # Show warning if truncated
        if total_noise > max_display:
            file_handle.write(f"""
                    <div class="warning-box">
                        <strong>⚠️ Display Limit Applied</strong><br>
                        Showing {max_display} of {total_noise} noise patterns.<br>
                        For complete list, please refer to: <code>clusters_final.txt</code>
                    </div>
""")
        
        # Apply display limit
        display_limit = min(max_display, len(noise_df))
        
        for _, row in noise_df.head(display_limit).iterrows():
            masked_url = str(row['masked'])
            original_count = int(row['original_count'])
            
            # Get original examples
            original_examples_str = row.get('original_examples', masked_url)
            if original_examples_str and '|||' in str(original_examples_str):
                original_examples = str(original_examples_str).split('|||')[:3]
            else:
                original_examples = [str(original_examples_str)] if original_examples_str else [masked_url]
            
            # CRITICAL FIX: Use html.escape()
            url_escaped = html.escape(masked_url, quote=True)
            
            file_handle.write(f"""
                    <div class="pattern-block noise-pattern">
                        <div class="pattern-header">
                            <div>
                                <strong>Pattern:</strong> <code class="url-pattern">{url_escaped}</code>
                            </div>
                            <span class="occurrence-badge">{original_count:,} occurrences</span>
                        </div>
                        <div class="original-examples">
                            <strong>🔍 Original URLs (for investigation):</strong>
                            <ul class="example-list">
""")
            
            for ex in original_examples:
                if ex and str(ex).strip():
                    ex_escaped = html.escape(str(ex), quote=True)
                    file_handle.write(f'                                <li>{ex_escaped}</li>\n')
            
            if original_count > len(original_examples):
                file_handle.write(f'                                <li><em>... and {original_count - len(original_examples)} more occurrences</em></li>\n')
            
            file_handle.write("""
                            </ul>
                        </div>
                    </div>
""")
        
        # Show more button if needed
        if total_noise > display_limit:
            file_handle.write(f"""
                    <div style="text-align: center; padding: 15px;">
                        <button class="show-more-btn" onclick="alert('Showing {display_limit} of {total_noise} noise patterns\\n\\nFull list available in:\\n- clusters_final.txt')">
                            📄 {total_noise - display_limit} more noise patterns (view in TXT)
                        </button>
                    </div>
""")
        
        file_handle.write("""
                </div>
            </div>
        </div>
""")
        print(f"     ✅ Noise section written ({display_limit} of {total_noise} shown)")
    
    def _get_html_footer(self):
        """Generate HTML footer with JavaScript"""
        return """
        <div class="footer">
            <p>Generated by URL Clustering Pipeline | Security Investigation Tool</p>
            <p style="margin-top: 5px; font-size: 0.85em;">
                For detailed analysis, refer to: clusters_final.csv & clusters_final.txt
            </p>
        </div>
    </div>
    
    <script>
        function toggleCluster(clusterId) {
            const body = document.getElementById('cluster-' + clusterId);
            if (body) {
                body.classList.toggle('active');
                const header = body.previousElementSibling;
                const arrow = header.querySelector('.cluster-count');
                if (arrow) {
                    arrow.textContent = body.classList.contains('active') ? '▲' : '▼';
                }
            }
        }
        
        // Initialize charts
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing charts...');
            
            // Pie Chart
            const pieCtx = document.getElementById('pieChart');
            if (pieCtx) {
                new Chart(pieCtx.getContext('2d'), {
                    type: 'pie',
                    data: {
                        labels: ['Clustered URLs', 'Noise URLs'],
                        datasets: [{
                            data: [chartData.pie.clustered, chartData.pie.noise],
                            backgroundColor: ['#4e73df', '#f6c23e'],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'bottom' },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const total = chartData.pie.clustered + chartData.pie.noise;
                                        const percentage = ((context.parsed / total) * 100).toFixed(1);
                                        return context.label + ': ' + context.parsed.toLocaleString() + ' (' + percentage + '%)';
                                    }
                                }
                            }
                        }
                    }
                });
            }
            
            // Bar Chart
            const barCtx = document.getElementById('barChart');
            if (barCtx) {
                const barLabels = chartData.bar.map(item => 'Cluster ' + item.cluster_id);
                const barData = chartData.bar.map(item => item.request_count);
                
                new Chart(barCtx.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: barLabels,
                        datasets: [{
                            label: 'Number of Requests',
                            data: barData,
                            backgroundColor: '#4e73df',
                            borderColor: '#2e59d9',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: {
                                    callback: function(value) {
                                        return value.toLocaleString();
                                    }
                                }
                            }
                        },
                        plugins: {
                            legend: { display: false }
                        }
                    }
                });
            }
            
            
            console.log('Charts initialized successfully!');
        });
    </script>
</body>
</html>
"""
    
    def step6_cleanup(self, keep_temp=False):
        """Step 6: Cleanup temporary files"""
        if keep_temp:
            print("\n✅ Temporary files kept in:", self.temp_dir)
            return True
        
        self.print_step(6, "CLEANUP")
        
        import shutil
        
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"✅ Removed temporary directory: {self.temp_dir}")
            
            if self.preprocessed_file.exists():
                self.preprocessed_file.unlink()
                print(f"✅ Removed preprocessed file: {self.preprocessed_file}")
            
            if self.original_urls_file.exists():
                self.original_urls_file.unlink()
                print(f"✅ Removed original URLs file: {self.original_urls_file}")
            
            return True
        except Exception as e:
            print(f"⚠️ Warning: Cleanup failed: {e}")
            return True
    
    def run(self, keep_temp=False):
        """Run complete pipeline"""
        self.start_time = time.time()
        
        self.print_header("WEB LOG CLUSTERING PIPELINE")
        print(f"Input: {self.log_file}")
        print(f"Output: {self.output_dir}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        steps = [
            ("Preprocessing", self.step1_preprocessing),
            ("Chunking", self.step2_chunking),
            ("Clustering", self.step3_clustering_chunks),
            ("Merging", self.step4_merge_results),
            ("HTML Report", self.step5_generate_html_report),
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"\n❌ Pipeline failed at: {step_name}")
                return False
        
        if not self.step6_cleanup(keep_temp):
            print("\n⚠️ Warning: Cleanup encountered issues (non-critical)")
        
        total_time = time.time() - self.start_time
        
        self.print_header("PIPELINE COMPLETED SUCCESSFULLY ✅")
        
        print("⏱️  Time Breakdown:")
        for step_name, step_time in self.step_times.items():
            percentage = (step_time / total_time) * 100
            print(f"  {step_name:.<30} {step_time:>8.2f}s ({percentage:>5.1f}%)")
        print(f"  {'─'*50}")
        print(f"  {'Total Time':.<30} {total_time:>8.2f}s ({total_time/60:>5.2f} min)")
        
        print(f"\n📁 Output Files:")
        print(f"  ✅ CSV Results    : {self.final_csv}")
        print(f"  ✅ Masked Patterns: {self.final_txt_masked}")
        print(f"  ✅ Original URLs  : {self.final_txt_original}")
        print(f"  ✅ HTML Report    : {self.output_dir / 'report.html'}")
        
        if keep_temp:
            print(f"  ℹ️  Temporary files: {self.temp_dir}")
        
        print(f"\n💡 Quick Access:")
        print(f"  View patterns: cat {self.final_txt_masked}")
        print(f"  View original URLs: cat {self.final_txt_original}")
        print(f"  Open HTML Report: open {self.output_dir / 'report.html'}")
        
        print("\n" + "="*80 + "\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete Web Log Clustering Pipeline (OPTIMIZED FOR LARGE DATASETS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_combined_fixed.py Sample-1k-Log.log outputs/
  python pipeline_combined_fixed.py large-log.log outputs/ --chunk-size 10000
  python pipeline_combined_fixed.py log.log outputs/ --keep-temp

OPTIMIZATIONS FOR LARGE DATASETS:
  ✓ Display limits to prevent browser crashes
  ✓ Proper HTML escaping (fixes rendering issues)
  ✓ Larger file buffers for faster writing
  ✓ Progress tracking for long operations
  ✓ Warning messages for very large files

Features:
  ✓ Full automation from log to report
  ✓ Interactive HTML with charts
  ✓ Handles 100k+ URLs efficiently
  ✓ Complete data in CSV/TXT files
        """
    )
    
    parser.add_argument("log_file", help="Input log file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=20000, help="URLs per chunk (default: 20000)")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"❌ Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    if os.path.getsize(args.log_file) == 0:
        print(f"❌ Error: Log file is empty")
        sys.exit(1)
    
    required_scripts = ["main2.py", "hdbscan_improved.py", "decoder.py"]
    missing_scripts = [s for s in required_scripts if not os.path.exists(s)]
    
    if missing_scripts:
        print(f"❌ Error: Required scripts not found: {', '.join(missing_scripts)}")
        sys.exit(1)
    
    chunk_size = args.chunk_size
    if chunk_size < 100 or chunk_size > 50000:
        print(f"⚠️  Warning: Unusual chunk size ({chunk_size})")
        print(f"   Recommended: 5000-30000")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    print("\n" + "="*80)
    print("PIPELINE CONFIGURATION")
    print("="*80)
    print(f"  Input file: {args.log_file}")
    print(f"  File size: {os.path.getsize(args.log_file):,} bytes")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Chunk size: {chunk_size:,} URLs")
    print("="*80 + "")
    
    orchestrator = PipelineOrchestrator(
        log_file=args.log_file,
        output_dir=args.output_dir,
        chunk_size=chunk_size
    )
    
    success = orchestrator.run(keep_temp=args.keep_temp)
    
    if success:
        print(" ✅ Success!")
        print("   1. Open report.html for interactive view")
        print("   2. Use CSV/TXT files for complete data\n")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()