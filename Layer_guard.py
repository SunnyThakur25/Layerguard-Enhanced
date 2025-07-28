#!/usr/bin/env python3
# layerguard_enhanced.py - Enhanced LLM Poison Detection Tool

import torch
import numpy as np
import json
import yaml
import hashlib
import sys
import os
from collections import defaultdict, Counter
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from datetime import datetime
import re
import traceback
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

# Try to import HuggingFace transformers
try:
    from transformers import PreTrainedModel, AutoModel, AutoConfig
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("⚠️  HuggingFace transformers not available. Install with: pip install transformers")

console = Console()

class LayerGuardEnhanced:
    def __init__(self, model_path, sensitivity='medium', layer_filter=None, batch_mode=False):
        self.model_path = model_path
        self.sensitivity = sensitivity
        self.layer_filter = layer_filter
        self.batch_mode = batch_mode
        self.model = None
        self.layers_data = {}
        self.poison_indicators = []
        self.findings = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'sensitivity': sensitivity,
            'layer_filter': layer_filter,
            'layer_analysis': {},
            'poisoned_layers': [],
            'risk_score': 0.0,
            'model_info': {}
        }
        
        # Sensitivity thresholds
        self.thresholds = {
            'low': {'z_score': 3.5, 'entropy_threshold': 0.1, 'outlier_contamination': 0.2},
            'medium': {'z_score': 3.0, 'entropy_threshold': 0.05, 'outlier_contamination': 0.15},
            'high': {'z_score': 2.5, 'entropy_threshold': 0.02, 'outlier_contamination': 0.1}
        }
        self.current_thresholds = self.thresholds[sensitivity]
    
    def load_model(self):
        """Load the LLM model with enhanced support"""
        try:
            console.print(f"[bold blue]Loading model from: {self.model_path}[/bold blue]")
            
            # Check if it's a HuggingFace model
            if HUGGINGFACE_AVAILABLE and os.path.isdir(self.model_path):
                console.print("[blue]Attempting HuggingFace model loading...[/blue]")
                try:
                    config = AutoConfig.from_pretrained(self.model_path)
                    self.model = AutoModel.from_pretrained(self.model_path, config=config)
                    self.findings['model_info'] = {
                        'type': 'huggingface',
                        'model_class': self.model.__class__.__name__,
                        'config': str(config.to_dict())[:200] + "..."
                    }
                    console.print(f"[green]✅ HuggingFace model loaded successfully![/green]")
                    return True
                except Exception as e:
                    console.print(f"[yellow]⚠️  HuggingFace loading failed: {str(e)}[/yellow]")
            
            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            console.print(f"[blue]Using device: {device}[/blue]")
            
            # Load model state dict
            if device == 'cuda':
                checkpoint = torch.load(self.model_path, map_location=device)
            else:
                checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Handle different model formats
            if isinstance(checkpoint, dict):
                # Standard PyTorch checkpoint
                self.model = checkpoint
                self.findings['model_info'] = {
                    'type': 'pytorch_checkpoint',
                    'keys': list(checkpoint.keys())[:10]  # First 10 keys
                }
            elif hasattr(checkpoint, 'state_dict'):
                # Model object with state dict
                self.model = checkpoint.state_dict()
                self.findings['model_info'] = {
                    'type': 'pytorch_model_object',
                    'model_class': checkpoint.__class__.__name__
                }
            else:
                # Direct state dict
                self.model = checkpoint
                self.findings['model_info'] = {
                    'type': 'pytorch_state_dict'
                }
            
            console.print(f"[green]✅ Model loaded successfully![/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error loading model: {str(e)}[/red]")
            console.print(f"[red]Full traceback: {traceback.format_exc()}[/red]")
            return False
    
    def should_analyze_layer(self, layer_name):
        """Check if layer should be analyzed based on filter"""
        if self.layer_filter is None:
            return True
        
        try:
            return bool(re.search(self.layer_filter, layer_name))
        except re.error:
            console.print(f"[red]❌ Invalid regex pattern: {self.layer_filter}[/red]")
            return True  # Analyze all if regex is invalid
    
    def extract_layer_statistics(self):
        """Extract comprehensive statistics from each layer"""
        console.print("[bold yellow]🔍 Extracting layer statistics...[/bold yellow]")
        
        if self.model is None:
            console.print("[red]❌ No model loaded[/red]")
            return False
        
        # Handle HuggingFace models
        if HUGGINGFACE_AVAILABLE and isinstance(self.model, PreTrainedModel):
            console.print("[blue]Processing HuggingFace model...[/blue]")
            model_state_dict = self.model.state_dict()
        else:
            model_state_dict = self.model
        
        if not isinstance(model_state_dict, dict):
            console.print("[red]❌ Model format not supported for analysis.[/red]")
            return False
        
        layer_stats = {}
        filtered_count = 0
        
        with Progress() as progress:
            # Count layers that match filter
            total_layers = sum(1 for name in model_state_dict.keys() if self.should_analyze_layer(name))
            task = progress.add_task("Analyzing layers...", total=total_layers)
            
            for layer_name, tensor in model_state_dict.items():
                # Apply layer filter
                if not self.should_analyze_layer(layer_name):
                    filtered_count += 1
                    continue
                
                try:
                    # Convert to numpy for easier analysis
                    if isinstance(tensor, torch.Tensor):
                        np_tensor = tensor.detach().cpu().numpy()
                    else:
                        np_tensor = np.array(tensor)
                    
                    # Basic statistics
                    stats_dict = {
                        'shape': np_tensor.shape,
                        'dtype': str(np_tensor.dtype),
                        'size': np_tensor.size,
                        'mean': float(np.mean(np_tensor)),
                        'std': float(np.std(np_tensor)),
                        'min': float(np.min(np_tensor)),
                        'max': float(np.max(np_tensor)),
                        'median': float(np.median(np_tensor)),
                        'entropy': self.calculate_differential_entropy(np_tensor),
                        'sparsity': self.calculate_sparsity(np_tensor),
                        'zero_ratio': float(np.sum(np_tensor == 0) / np_tensor.size),
                        'positive_ratio': float(np.sum(np_tensor > 0) / np_tensor.size),
                        'negative_ratio': float(np.sum(np_tensor < 0) / np_tensor.size),
                        'unique_values': min(len(np.unique(np_tensor.flatten()[:10000])), 10000),  # Limit for large tensors
                        'norm_l1': float(np.linalg.norm(np_tensor.flatten(), ord=1)),
                        'norm_l2': float(np.linalg.norm(np_tensor.flatten(), ord=2)),
                        'kurtosis': float(stats.kurtosis(np_tensor.flatten()[:100000])),  # Sample for large tensors
                        'skewness': float(stats.skew(np_tensor.flatten()[:100000]))
                    }
                    
                    layer_stats[layer_name] = stats_dict
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    console.print(f"[red]Error analyzing layer {layer_name}: {str(e)}[/red]")
                    progress.update(task, advance=1)
                    continue
        
        self.layers_data = layer_stats
        console.print(f"[green]✅ Analyzed {len(layer_stats)} layers[/green]")
        if filtered_count > 0:
            console.print(f"[blue]ℹ️  Filtered out {filtered_count} layers[/blue]")
        return True
    
    def calculate_differential_entropy(self, tensor):
        """Calculate differential entropy for continuous weights"""
        try:
            # Flatten and sample if too large
            flat_tensor = tensor.flatten()
            
            # For very large tensors, sample
            if len(flat_tensor) > 100000:
                indices = np.random.choice(len(flat_tensor), 100000, replace=False)
                flat_tensor = flat_tensor[indices]
            
            # Estimate probability density using kernel density estimation
            from scipy.stats import gaussian_kde
            
            # Handle constant values
            if np.allclose(flat_tensor, flat_tensor[0]):
                return 0.0
            
            try:
                kde = gaussian_kde(flat_tensor)
                # Evaluate KDE on a grid
                x_grid = np.linspace(flat_tensor.min(), flat_tensor.max(), 1000)
                pdf = kde(x_grid)
                
                # Remove zeros to avoid log(0)
                pdf = pdf[pdf > 1e-10]
                
                # Calculate differential entropy: -∫ p(x) log p(x) dx
                dx = x_grid[1] - x_grid[0]
                diff_entropy = -np.sum(pdf * np.log(pdf)) * dx
                
                return float(diff_entropy)
            except Exception:
                # Fallback to discrete entropy
                return self.calculate_discrete_entropy(flat_tensor)
                
        except Exception:
            return 0.0
    
    def calculate_discrete_entropy(self, tensor):
        """Calculate discrete entropy as fallback"""
        try:
            # Bin the continuous values
            hist, _ = np.histogram(tensor, bins=100, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropy = -np.sum(hist * np.log(hist + 1e-10))  # Add small epsilon
            return float(entropy)
        except:
            return 0.0
    
    def calculate_sparsity(self, tensor):
        """Calculate sparsity of tensor"""
        try:
            # Count near-zero elements
            near_zero = np.sum(np.abs(tensor) < 1e-6)
            return float(near_zero / tensor.size)
        except:
            return 0.0
    
    def detect_statistical_anomalies(self):
        """Detect statistical anomalies in layer parameters"""
        console.print("[bold yellow]🔍 Detecting statistical anomalies...[/bold yellow]")
        
        if not self.layers_data:
            console.print("[red]❌ No layer data available[/red]")
            return
        
        # Collect numerical features for anomaly detection
        features = []
        layer_names = []
        
        numerical_features = [
            'mean', 'std', 'min', 'max', 'median', 'entropy', 
            'sparsity', 'zero_ratio', 'positive_ratio', 'negative_ratio',
            'norm_l1', 'norm_l2', 'kurtosis', 'skewness'
        ]
        
        for layer_name, stats in self.layers_data.items():
            feature_vector = []
            valid = True
            
            for feature in numerical_features:
                if feature in stats:
                    feature_vector.append(stats[feature])
                else:
                    feature_vector.append(0.0)
                    valid = False
            
            if valid and len(feature_vector) == len(numerical_features):
                features.append(feature_vector)
                layer_names.append(layer_name)
        
        if not features:
            console.print("[red]❌ No valid features for anomaly detection[/red]")
            return
        
        # Convert to numpy array
        X = np.array(features)
        
        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest for outlier detection
        iso_forest = IsolationForest(
            contamination=self.current_thresholds['outlier_contamination'],
            random_state=42
        )
        outlier_predictions = iso_forest.fit_predict(X_scaled)
        outlier_scores = iso_forest.decision_function(X_scaled)
        
        # Statistical Z-score analysis
        z_scores = np.abs(stats.zscore(X_scaled, axis=0))
        max_z_scores = np.max(z_scores, axis=1)
        
        # Identify poisoned layers
        poisoned_layers = []
        
        for i, (layer_name, stats) in enumerate(zip(layer_names, features)):
            layer_poison_indicators = []
            
            # Check if marked as outlier by Isolation Forest
            if outlier_predictions[i] == -1:
                layer_poison_indicators.append({
                    'type': 'isolation_forest_outlier',
                    'score': float(outlier_scores[i]),
                    'severity': 'high' if outlier_scores[i] < -0.5 else 'medium'
                })
            
            # Check Z-score anomalies
            if max_z_scores[i] > self.current_thresholds['z_score']:
                layer_poison_indicators.append({
                    'type': 'z_score_anomaly',
                    'score': float(max_z_scores[i]),
                    'severity': 'high' if max_z_scores[i] > self.current_thresholds['z_score'] * 1.5 else 'medium'
                })
            
            # Check entropy anomalies (very low entropy might indicate poisoning)
            if 'entropy' in self.layers_data[layer_name] and \
               self.layers_data[layer_name]['entropy'] < self.current_thresholds['entropy_threshold']:
                layer_poison_indicators.append({
                    'type': 'low_entropy',
                    'score': float(self.layers_data[layer_name]['entropy']),
                    'severity': 'high'
                })
            
            # Check for suspicious value patterns
            suspicious_patterns = self.detect_suspicious_patterns(layer_name, self.layers_data[layer_name])
            layer_poison_indicators.extend(suspicious_patterns)
            
            if layer_poison_indicators:
                poisoned_layers.append({
                    'layer_name': layer_name,
                    'indicators': layer_poison_indicators,
                    'risk_score': self.calculate_layer_risk_score(layer_poison_indicators),
                    'stats': self.layers_data[layer_name]
                })
        
        # Sort by risk score
        poisoned_layers.sort(key=lambda x: x['risk_score'], reverse=True)
        
        self.findings['poisoned_layers'] = poisoned_layers
        self.findings['risk_score'] = self.calculate_overall_risk_score(poisoned_layers)
        
        console.print(f"[green]✅ Analysis complete. Found {len(poisoned_layers)} potentially poisoned layers[/green]")
    
    def detect_suspicious_patterns(self, layer_name, layer_stats):
        """Detect suspicious patterns that might indicate poisoning"""
        indicators = []
        
        # Check for extreme values
        if abs(layer_stats.get('mean', 0)) > 100:
            indicators.append({
                'type': 'extreme_mean',
                'score': abs(layer_stats['mean']),
                'severity': 'high'
            })
        
        # Check for very high standard deviation
        if layer_stats.get('std', 0) > 100:
            indicators.append({
                'type': 'extreme_std',
                'score': layer_stats['std'],
                'severity': 'high'
            })
        
        # Check for suspicious zero ratios
        if layer_stats.get('zero_ratio', 0) > 0.9:
            indicators.append({
                'type': 'excessive_zeros',
                'score': layer_stats['zero_ratio'],
                'severity': 'medium'
            })
        
        # Check for uniform values (low diversity)
        if layer_stats.get('unique_values', 10000) < 10:
            indicators.append({
                'type': 'low_diversity',
                'score': layer_stats['unique_values'],
                'severity': 'high'
            })
        
        # Check for layer name patterns that might indicate backdoors
        suspicious_keywords = ['trigger', 'backdoor', 'trojan', 'poison', 'malicious', 'hidden']
        for keyword in suspicious_keywords:
            if keyword in layer_name.lower():
                indicators.append({
                    'type': 'suspicious_layer_name',
                    'score': 1.0,
                    'severity': 'high'
                })
                break
        
        # Check for statistical anomalies
        if abs(layer_stats.get('skewness', 0)) > 10:
            indicators.append({
                'type': 'extreme_skewness',
                'score': abs(layer_stats['skewness']),
                'severity': 'medium'
            })
        
        if abs(layer_stats.get('kurtosis', 0)) > 100:
            indicators.append({
                'type': 'extreme_kurtosis',
                'score': abs(layer_stats['kurtosis']),
                'severity': 'medium'
            })
        
        return indicators
    
    def calculate_layer_risk_score(self, indicators):
        """Calculate risk score for a layer based on indicators"""
        score = 0.0
        weights = {'high': 3.0, 'medium': 1.5, 'low': 0.5}
        
        for indicator in indicators:
            severity = indicator.get('severity', 'low')
            score += weights.get(severity, 0.5)
        
        return min(score, 10.0)  # Cap at 10
    
    def calculate_overall_risk_score(self, poisoned_layers):
        """Calculate overall model risk score"""
        if not poisoned_layers:
            return 0.0
        
        total_score = sum(layer['risk_score'] for layer in poisoned_layers)
        avg_score = total_score / len(poisoned_layers)
        
        # Weight by number of poisoned layers
        layer_ratio = min(len(poisoned_layers) / max(len(self.layers_data), 1), 1.0)
        
        return min(avg_score * (1 + layer_ratio), 10.0)
    
    def generate_report(self, output_file=None, format='json'):
        """Generate comprehensive analysis report in multiple formats"""
        console.print(f"[bold blue]📋 Generating analysis report ({format})...[/bold blue]")
        
        # Create rich table for summary
        table = Table(title="LayerGuard Enhanced Poison Detection Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Model Path", self.findings['model_path'])
        table.add_row("Model Type", self.findings['model_info'].get('type', 'unknown'))
        table.add_row("Layers Analyzed", str(len(self.layers_data)))
        table.add_row("Poisoned Layers", str(len(self.findings['poisoned_layers'])))
        table.add_row("Overall Risk Score", f"{self.findings['risk_score']:.2f}/10.0")
        table.add_row("Sensitivity Level", self.findings['sensitivity'])
        table.add_row("Layer Filter", str(self.findings['layer_filter']))
        table.add_row("Analysis Timestamp", self.findings['timestamp'])
        
        console.print(table)
        
        # Show top poisoned layers
        if self.findings['poisoned_layers']:
            console.print("\n[bold red]🚨 Top Poisoned Layers:[/bold red]")
            layer_table = Table()
            layer_table.add_column("Layer Name", style="yellow")
            layer_table.add_column("Risk Score", style="red")
            layer_table.add_column("Indicators", style="blue")
            
            for layer in self.findings['poisoned_layers'][:10]:  # Top 10
                indicators = ", ".join([ind['type'] for ind in layer['indicators']])
                layer_table.add_row(
                    layer['layer_name'][:50] + "..." if len(layer['layer_name']) > 50 else layer['layer_name'],
                    f"{layer['risk_score']:.2f}",
                    indicators[:50] + "..." if len(indicators) > 50 else indicators
                )
            
            console.print(layer_table)
        
        # Save detailed report in requested format
        if output_file:
            if format == 'json':
                with open(output_file, 'w') as f:
                    json.dump(self.findings, f, indent=2)
                console.print(f"[green]✅ Detailed JSON report saved to: {output_file}[/green]")
            elif format == 'yaml':
                with open(output_file, 'w') as f:
                    yaml.dump(self.findings, f, default_flow_style=False, indent=2)
                console.print(f"[green]✅ Detailed YAML report saved to: {output_file}[/green]")
            elif format == 'markdown':
                self._save_markdown_report(output_file)
                console.print(f"[green]✅ Detailed Markdown report saved to: {output_file}[/green]")
        
        return self.findings
    
    def _save_markdown_report(self, output_file):
        """Save report in human-readable Markdown format"""
        with open(output_file, 'w') as f:
            f.write(f"# LayerGuard Enhanced Analysis Report\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **Model Path**: {self.findings['model_path']}\n")
            f.write(f"- **Model Type**: {self.findings['model_info'].get('type', 'unknown')}\n")
            f.write(f"- **Layers Analyzed**: {len(self.layers_data)}\n")
            f.write(f"- **Poisoned Layers**: {len(self.findings['poisoned_layers'])}\n")
            f.write(f"- **Overall Risk Score**: {self.findings['risk_score']:.2f}/10.0\n")
            f.write(f"- **Sensitivity Level**: {self.findings['sensitivity']}\n")
            f.write(f"- **Analysis Timestamp**: {self.findings['timestamp']}\n\n")
            
            if self.findings['poisoned_layers']:
                f.write(f"## Top Poisoned Layers\n\n")
                f.write("| Layer Name | Risk Score | Indicators |\n")
                f.write("|------------|------------|------------|\n")
                
                for layer in self.findings['poisoned_layers'][:20]:
                    indicators = ", ".join([ind['type'] for ind in layer['indicators']])
                    layer_name = layer['layer_name'][:50] + "..." if len(layer['layer_name']) > 50 else layer['layer_name']
                    f.write(f"| {layer_name} | {layer['risk_score']:.2f} | {indicators} |\n")
    
    def visualize_results(self, output_dir="layerguard_visualizations"):
        """Create visualizations of the analysis results in multiple formats"""
        console.print("[bold yellow]📊 Creating visualizations...[/bold yellow]")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract numerical data for visualization
        if not self.layers_data:
            console.print("[red]❌ No data to visualize[/red]")
            return
        
        # Create statistics plots
        stats_to_plot = ['mean', 'std', 'entropy', 'sparsity', 'kurtosis', 'skewness']
        data_for_plotting = defaultdict(list)
        
        for layer_name, stats in self.layers_data.items():
            for stat in stats_to_plot:
                if stat in stats:
                    data_for_plotting[stat].append(stats[stat])
        
        # Plot distributions
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LLM Layer Statistics Distributions', fontsize=16)
        
        for i, stat in enumerate(stats_to_plot):
            row, col = i // 3, i % 3
            if data_for_plotting[stat]:
                # Remove outliers for better visualization
                data = np.array(data_for_plotting[stat])
                q75, q25 = np.percentile(data, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - (iqr * 1.5)
                upper_bound = q75 + (iqr * 1.5)
                filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
                
                axes[row, col].hist(filtered_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
                axes[row, col].set_title(f'{stat.capitalize()} Distribution')
                axes[row, col].set_xlabel(stat)
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_statistics.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'layer_statistics.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot risk scores
        if self.findings['poisoned_layers']:
            risk_scores = [layer['risk_score'] for layer in self.findings['poisoned_layers']]
            layer_names = [layer['layer_name'] for layer in self.findings['poisoned_layers']]
            
            plt.figure(figsize=(15, 10))
            bars = plt.barh(range(len(risk_scores[:20])), risk_scores[:20], color='coral')  # Top 20
            plt.yticks(range(len(risk_scores[:20])), 
                      [name[:40] + "..." if len(name) > 40 else name for name in layer_names[:20]],
                      fontsize=8)
            plt.xlabel('Risk Score', fontsize=12)
            plt.ylabel('Layer Name', fontsize=12)
            plt.title('Top 20 Poisoned Layer Risk Scores', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'poisoned_layers_risk.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'poisoned_layers_risk.pdf'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create correlation heatmap
        if len(self.layers_data) > 1:
            # Sample data for correlation (avoid memory issues with large models)
            sample_size = min(100, len(self.layers_data))
            sampled_layers = dict(list(self.layers_data.items())[:sample_size])
            
            correlation_data = []
            for layer_name, stats in sampled_layers.items():
                row = [stats.get('mean', 0), stats.get('std', 0), stats.get('entropy', 0), 
                       stats.get('sparsity', 0), stats.get('kurtosis', 0), stats.get('skewness', 0)]
                correlation_data.append(row)
            
            if correlation_data:
                df_corr = np.array(correlation_data)
                corr_matrix = np.corrcoef(df_corr.T)
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           xticklabels=['Mean', 'Std', 'Entropy', 'Sparsity', 'Kurtosis', 'Skewness'],
                           yticklabels=['Mean', 'Std', 'Entropy', 'Sparsity', 'Kurtosis', 'Skewness'])
                plt.title('Layer Statistics Correlation Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(output_dir, 'correlation_heatmap.pdf'), dpi=300, bbox_inches='tight')
                plt.close()
        
        console.print(f"[green]✅ Visualizations saved to: {output_dir} (PNG and PDF formats)[/green]")
    
    def run_complete_analysis(self, output_file=None, format='json', visualize=False):
        """Run complete poison detection analysis"""
        console.print("[bold blue]🚀 Starting Enhanced LayerGuard Poison Detection Analysis[/bold blue]")
        console.print(f"[blue]Model: {self.model_path}[/blue]")
        console.print(f"[blue]Sensitivity: {self.sensitivity}[/blue]")
        if self.layer_filter:
            console.print(f"[blue]Layer Filter: {self.layer_filter}[/blue]")
        
        # Step 1: Load model
        if not self.load_model():
            return None
        
        # Step 2: Extract layer statistics
        if not self.extract_layer_statistics():
            return None
        
        # Step 3: Detect anomalies
        self.detect_statistical_anomalies()
        
        # Step 4: Generate report
        findings = self.generate_report(output_file, format)
        
        # Step 5: Create visualizations
        if visualize:
            self.visualize_results()
        
        console.print("[bold green]✅ Analysis complete![/bold green]")
        return findings

class BatchProcessor:
    """Handle batch processing of multiple models"""
    
    def __init__(self, model_paths, sensitivity='medium', layer_filter=None):
        self.model_paths = model_paths
        self.sensitivity = sensitivity
        self.layer_filter = layer_filter
        self.results = []
        self.batch_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(model_paths),
            'processed_models': 0,
            'high_risk_models': 0,
            'medium_risk_models': 0,
            'low_risk_models': 0,
            'failed_models': 0
        }
    
    def process_models(self, output_dir="batch_reports", visualize=False):
        """Process all models in batch"""
        console.print(f"[bold blue]🔄 Starting batch processing of {len(self.model_paths)} models[/bold blue]")
        
        os.makedirs(output_dir, exist_ok=True)
        
        with Progress() as progress:
            task = progress.add_task("Processing models...", total=len(self.model_paths))
            
            for i, model_path in enumerate(self.model_paths):
                progress.set_description(f"Processing {os.path.basename(model_path)}")
                
                try:
                    # Create individual detector
                    detector = LayerGuardEnhanced(
                        model_path, 
                        self.sensitivity, 
                        self.layer_filter,
                        batch_mode=True
                    )
                    
                    # Run analysis
                    findings = detector.run_complete_analysis(
                        output_file=os.path.join(output_dir, f"report_{i}_{os.path.basename(model_path)}.json"),
                        visualize=visualize
                    )
                    
                    if findings:
                        self.results.append({
                            'model_path': model_path,
                            'findings': findings,
                            'risk_score': findings['risk_score']
                        })
                        
                        # Update batch summary
                        self.batch_summary['processed_models'] += 1
                        if findings['risk_score'] > 7.0:
                            self.batch_summary['high_risk_models'] += 1
                        elif findings['risk_score'] > 4.0:
                            self.batch_summary['medium_risk_models'] += 1
                        else:
                            self.batch_summary['low_risk_models'] += 1
                    else:
                        self.batch_summary['failed_models'] += 1
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed to process {model_path}: {str(e)}[/red]")
                    self.batch_summary['failed_models'] += 1
                
                progress.update(task, advance=1)
        
        # Generate batch summary report
        self.generate_batch_report(output_dir)
        return self.results
    
    def generate_batch_report(self, output_dir):
        """Generate batch processing summary report"""
        batch_report = {
            'batch_summary': self.batch_summary,
            'model_results': []
        }
        
        for result in self.results:
            batch_report['model_results'].append({
                'model_path': result['model_path'],
                'risk_score': result['risk_score'],
                'poisoned_layers': len(result['findings']['poisoned_layers'])
            })
        
        # Sort by risk score
        batch_report['model_results'].sort(key=lambda x: x['risk_score'], reverse=True)
        
        # Save batch report
        report_path = os.path.join(output_dir, "batch_summary.json")
        with open(report_path, 'w') as f:
            json.dump(batch_report, f, indent=2)
        
        # Also save as markdown
        md_report_path = os.path.join(output_dir, "batch_summary.md")
        with open(md_report_path, 'w') as f:
            f.write("# LayerGuard Batch Processing Report\n\n")
            f.write("## Summary\n\n")
            f.write(f"- **Total Models**: {self.batch_summary['total_models']}\n")
            f.write(f"- **Processed**: {self.batch_summary['processed_models']}\n")
            f.write(f"- **High Risk**: {self.batch_summary['high_risk_models']}\n")
            f.write(f"- **Medium Risk**: {self.batch_summary['medium_risk_models']}\n")
            f.write(f"- **Low Risk**: {self.batch_summary['low_risk_models']}\n")
            f.write(f"- **Failed**: {self.batch_summary['failed_models']}\n\n")
            
            f.write("## Model Rankings\n\n")
            f.write("| Rank | Model | Risk Score | Poisoned Layers |\n")
            f.write("|------|-------|------------|-----------------|\n")
            
            for i, result in enumerate(batch_report['model_results'][:20]):  # Top 20
                model_name = os.path.basename(result['model_path'])
                f.write(f"| {i+1} | {model_name} | {result['risk_score']:.2f} | {result['poisoned_layers']} |\n")
        
        console.print(f"[green]✅ Batch processing complete. Reports saved to: {output_dir}[/green]")

@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--sensitivity', '-s', default='medium', 
              type=click.Choice(['low', 'medium', 'high']), 
              help='Detection sensitivity level')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for detailed report')
@click.option('--format', '-f', default='json', 
              type=click.Choice(['json', 'yaml', 'markdown']), 
              help='Report format')
@click.option('--visualize', '-v', is_flag=True, 
              help='Generate visualization charts (PNG and PDF)')
@click.option('--hash-check', '-c', is_flag=True,
              help='Perform integrity hash check')
@click.option('--layer-filter', '-l', type=str,
              help='Regex pattern to filter layers for analysis')
@click.option('--batch-process', '-b', type=click.Path(exists=True),
              help='Process multiple models from file list (one path per line)')
def main(model_path, sensitivity, output, format, visualize, hash_check, layer_filter, batch_process):
    """Enhanced LayerGuard: Advanced LLM Poison Detection Tool
    
    Detects poisoned weight layers in Large Language Models using 
    statistical analysis, anomaly detection, and pattern recognition.
    
    Supports HuggingFace Transformers, layer filtering, enhanced entropy,
    multiple report formats, and batch processing.
    
    MODEL_PATH: Path to the LLM model file (.pth, .pt, .bin, etc.) or directory for HuggingFace models
    """
    
    # Handle batch processing
    if batch_process:
        console.print("[bold blue]🔄 Batch processing mode activated[/bold blue]")
        
        # Read model paths from file
        try:
            with open(batch_process, 'r') as f:
                model_paths = [line.strip() for line in f if line.strip()]
            
            if not model_paths:
                console.print("[red]❌ No model paths found in batch file[/red]")
                sys.exit(1)
            
            # Process batch
            batch_processor = BatchProcessor(model_paths, sensitivity, layer_filter)
            results = batch_processor.process_models(visualize=visualize)
            
            # Exit based on batch results
            high_risk = sum(1 for r in results if r['risk_score'] > 7.0)
            medium_risk = sum(1 for r in results if 4.0 < r['risk_score'] <= 7.0)
            
            if high_risk > 0:
                console.print(f"[bold red]🚨 {high_risk} HIGH RISK models detected in batch[/bold red]")
                sys.exit(2)
            elif medium_risk > 0:
                console.print(f"[bold yellow]⚠️  {medium_risk} MEDIUM RISK models detected in batch[/bold yellow]")
                sys.exit(1)
            else:
                console.print("[bold green]✅ All models in batch passed inspection[/bold green]")
                sys.exit(0)
                
        except Exception as e:
            console.print(f"[red]❌ Batch processing failed: {str(e)}[/red]")
            sys.exit(1)
    
    # Perform hash check if requested
    if hash_check:
        console.print("[bold yellow]🔍 Performing integrity check...[/bold yellow]")
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        console.print(f"[blue]File MD5 Hash: {hash_md5.hexdigest()}[/blue]")
    
    # Run single model analysis
    detector = LayerGuardEnhanced(model_path, sensitivity, layer_filter)
    findings = detector.run_complete_analysis(output, format, visualize)
    
    if findings is None:
        sys.exit(1)
    
    # Exit with appropriate code based on risk
    if findings['risk_score'] > 7.0:
        console.print("[bold red]🚨 HIGH RISK: Model shows significant poisoning indicators[/bold red]")
        sys.exit(2)
    elif findings['risk_score'] > 4.0:
        console.print("[bold yellow]⚠️  MEDIUM RISK: Model shows some suspicious indicators[/bold yellow]")
        sys.exit(1)
    else:
        console.print("[bold green]✅ LOW RISK: No significant poisoning indicators detected[/bold green]")
        sys.exit(0)

if __name__ == '__main__':
    main()