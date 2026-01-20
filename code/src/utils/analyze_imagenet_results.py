#!/usr/bin/env python3
"""
ImageNet Analysis and Results Script
Analyzes evaluation results and creates comprehensive reports
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ImageNetAnalyzer:
    def __init__(self, results_file='imagenet_pretrained_evaluation.json'):
        self.results_file = results_file
        self.results = None
        self.load_results()
    
    def load_results(self):
        """Load evaluation results from JSON file"""
        if not os.path.exists(self.results_file):
            print(f"❌ Results file not found: {self.results_file}")
            print("Please run the evaluation script first.")
            return False
        
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            print(f"✅ Loaded results from {self.results_file}")
            return True
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def analyze_metrics(self):
        """Analyze and compare metrics between models"""
        if not self.results:
            return
        
        print("\n" + "="*80)
        print("IMAGENET EVALUATION ANALYSIS")
        print("="*80)
        
        # Extract metrics for both models
        eccv16_data = self.results.get('eccv16_pretrained', {})
        siggraph17_data = self.results.get('siggraph17_pretrained', {})
        
        if not eccv16_data or not siggraph17_data:
            print("❌ Missing model data in results")
            return
        
        eccv16_summary = eccv16_data.get('summary', {})
        siggraph17_summary = siggraph17_data.get('summary', {})
        
        # Print detailed analysis
        print("\n📊 DETAILED METRICS ANALYSIS")
        print("-" * 60)
        
        metrics = ['psnr', 'ssim', 'color_diff', 'inference_times']
        metric_names = ['PSNR', 'SSIM', 'Color Difference', 'Inference Time (ms)']
        
        for metric, name in zip(metrics, metric_names):
            if metric in eccv16_summary and metric in siggraph17_summary:
                eccv16_val = eccv16_summary[metric]['mean']
                siggraph17_val = siggraph17_summary[metric]['mean']
                
                print(f"\n{name}:")
                print(f"  ECCV16:     {eccv16_val:.3f} ± {eccv16_summary[metric]['std']:.3f}")
                print(f"  SIGGRAPH17: {siggraph17_val:.3f} ± {siggraph17_summary[metric]['std']:.3f}")
                
                if metric == 'color_diff':
                    # Lower is better for color difference
                    better_model = "ECCV16" if eccv16_val < siggraph17_val else "SIGGRAPH17"
                    improvement = abs(eccv16_val - siggraph17_val) / max(eccv16_val, siggraph17_val) * 100
                else:
                    # Higher is better for PSNR, SSIM, lower is better for inference time
                    if metric == 'inference_times':
                        better_model = "ECCV16" if eccv16_val < siggraph17_val else "SIGGRAPH17"
                    else:
                        better_model = "ECCV16" if eccv16_val > siggraph17_val else "SIGGRAPH17"
                    improvement = abs(eccv16_val - siggraph17_val) / min(eccv16_val, siggraph17_val) * 100
                
                print(f"  Better: {better_model} ({improvement:.1f}% difference)")
        
        # Success rate analysis
        print(f"\n📈 SUCCESS RATE ANALYSIS")
        print("-" * 40)
        eccv16_success = eccv16_summary.get('success_rate', 0)
        siggraph17_success = siggraph17_summary.get('success_rate', 0)
        
        print(f"ECCV16 Success Rate:     {eccv16_success:.3f} ({eccv16_summary.get('successful_predictions', 0)}/{eccv16_summary.get('total_samples', 0)})")
        print(f"SIGGRAPH17 Success Rate: {siggraph17_success:.3f} ({siggraph17_summary.get('successful_predictions', 0)}/{siggraph17_summary.get('total_samples', 0)})")
    
    def create_metrics_visualization(self):
        """Create visualization of metrics comparison"""
        if not self.results:
            return
        
        print("\n🎨 Creating metrics visualization...")
        
        # Extract data
        eccv16_summary = self.results.get('eccv16_pretrained', {}).get('summary', {})
        siggraph17_summary = self.results.get('siggraph17_pretrained', {}).get('summary', {})
        
        if not eccv16_summary or not siggraph17_summary:
            print("❌ Missing summary data")
            return
        
        # Prepare data for plotting
        metrics = ['psnr', 'ssim', 'color_diff']
        metric_names = ['PSNR', 'SSIM', 'Color Difference']
        
        eccv16_values = []
        siggraph17_values = []
        eccv16_stds = []
        siggraph17_stds = []
        
        for metric in metrics:
            if metric in eccv16_summary and metric in siggraph17_summary:
                eccv16_values.append(eccv16_summary[metric]['mean'])
                siggraph17_values.append(siggraph17_summary[metric]['mean'])
                eccv16_stds.append(eccv16_summary[metric]['std'])
                siggraph17_stds.append(siggraph17_summary[metric]['std'])
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ImageNet Pretrained Models Comparison', fontsize=16, fontweight='bold')
        
        # Bar plot comparison
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, eccv16_values, width, label='ECCV16', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, siggraph17_values, width, label='SIGGRAPH17', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Values')
        axes[0, 0].set_title('Metrics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metric_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error bars
        axes[0, 0].errorbar(x - width/2, eccv16_values, yerr=eccv16_stds, fmt='none', color='black', capsize=5)
        axes[0, 0].errorbar(x + width/2, siggraph17_values, yerr=siggraph17_stds, fmt='none', color='black', capsize=5)
        
        # Inference time comparison
        inference_times = [eccv16_summary.get('inference_times', {}).get('mean', 0),
                          siggraph17_summary.get('inference_times', {}).get('mean', 0)]
        inference_stds = [eccv16_summary.get('inference_times', {}).get('std', 0),
                          siggraph17_summary.get('inference_times', {}).get('std', 0)]
        
        axes[0, 1].bar(['ECCV16', 'SIGGRAPH17'], inference_times, alpha=0.8, 
                      color=['skyblue', 'lightcoral'])
        axes[0, 1].errorbar(['ECCV16', 'SIGGRAPH17'], inference_times, yerr=inference_stds, 
                           fmt='none', color='black', capsize=5)
        axes[0, 1].set_ylabel('Inference Time (ms)')
        axes[0, 1].set_title('Inference Time Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate comparison
        success_rates = [eccv16_summary.get('success_rate', 0),
                         siggraph17_summary.get('success_rate', 0)]
        
        axes[1, 0].bar(['ECCV16', 'SIGGRAPH17'], success_rates, alpha=0.8,
                      color=['skyblue', 'lightcoral'])
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title('Success Rate Comparison')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Radar chart for overall performance
        categories = ['PSNR', 'SSIM', 'Speed', 'Success Rate']
        
        # Normalize values for radar chart (0-1 scale)
        eccv16_normalized = [
            eccv16_summary.get('psnr', {}).get('mean', 0) / 50,  # Normalize PSNR
            eccv16_summary.get('ssim', {}).get('mean', 0),        # SSIM already 0-1
            1 - (eccv16_summary.get('inference_times', {}).get('mean', 0) / 1000),  # Normalize speed
            eccv16_summary.get('success_rate', 0)                # Success rate already 0-1
        ]
        
        siggraph17_normalized = [
            siggraph17_summary.get('psnr', {}).get('mean', 0) / 50,
            siggraph17_summary.get('ssim', {}).get('mean', 0),
            1 - (siggraph17_summary.get('inference_times', {}).get('mean', 0) / 1000),
            siggraph17_summary.get('success_rate', 0)
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        eccv16_normalized += eccv16_normalized[:1]
        siggraph17_normalized += siggraph17_normalized[:1]
        
        axes[1, 1].plot(angles, eccv16_normalized, 'o-', linewidth=2, label='ECCV16', color='skyblue')
        axes[1, 1].fill(angles, eccv16_normalized, alpha=0.25, color='skyblue')
        axes[1, 1].plot(angles, siggraph17_normalized, 'o-', linewidth=2, label='SIGGRAPH17', color='lightcoral')
        axes[1, 1].fill(angles, siggraph17_normalized, alpha=0.25, color='lightcoral')
        
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Overall Performance Radar')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('imagenet_metrics_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Metrics visualization saved as 'imagenet_metrics_analysis.png'")
    
    def create_detailed_report(self):
        """Create a detailed text report"""
        if not self.results:
            return
        
        print("\n📄 Creating detailed report...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("IMAGENET PRETRAINED MODELS EVALUATION REPORT")
        report_lines.append("="*80)
        
        # Metadata
        metadata = self.results.get('metadata', {})
        report_lines.append(f"\nDataset: {metadata.get('dataset', 'ImageNet')}")
        report_lines.append(f"Evaluation Date: {metadata.get('evaluation_date', 'Unknown')}")
        report_lines.append(f"Device: {metadata.get('device', 'Unknown')}")
        report_lines.append(f"Models Evaluated: {', '.join(metadata.get('models_evaluated', []))}")
        
        # Model comparisons
        for model_name in ['eccv16_pretrained', 'siggraph17_pretrained']:
            if model_name in self.results:
                model_data = self.results[model_name]
                summary = model_data.get('summary', {})
                
                report_lines.append(f"\n{'-'*60}")
                report_lines.append(f"{model_name.upper().replace('_', ' ')} RESULTS")
                report_lines.append(f"{'-'*60}")
                
                report_lines.append(f"Total Samples: {summary.get('total_samples', 0)}")
                report_lines.append(f"Successful Predictions: {summary.get('successful_predictions', 0)}")
                report_lines.append(f"Success Rate: {summary.get('success_rate', 0):.3f}")
                
                for metric in ['psnr', 'ssim', 'color_diff', 'inference_times']:
                    if metric in summary:
                        metric_data = summary[metric]
                        report_lines.append(f"{metric.upper()}:")
                        report_lines.append(f"  Mean: {metric_data.get('mean', 0):.3f}")
                        report_lines.append(f"  Std:  {metric_data.get('std', 0):.3f}")
                        report_lines.append(f"  Min:  {metric_data.get('min', 0):.3f}")
                        report_lines.append(f"  Max:  {metric_data.get('max', 0):.3f}")
        
        # Recommendations
        report_lines.append(f"\n{'-'*60}")
        report_lines.append("RECOMMENDATIONS")
        report_lines.append(f"{'-'*60}")
        
        eccv16_summary = self.results.get('eccv16_pretrained', {}).get('summary', {})
        siggraph17_summary = self.results.get('siggraph17_pretrained', {}).get('summary', {})
        
        if eccv16_summary and siggraph17_summary:
            # Compare models
            eccv16_psnr = eccv16_summary.get('psnr', {}).get('mean', 0)
            siggraph17_psnr = siggraph17_summary.get('psnr', {}).get('mean', 0)
            
            eccv16_speed = eccv16_summary.get('inference_times', {}).get('mean', 0)
            siggraph17_speed = siggraph17_summary.get('inference_times', {}).get('mean', 0)
            
            if eccv16_psnr > siggraph17_psnr:
                report_lines.append("• ECCV16 shows better reconstruction quality (higher PSNR)")
            else:
                report_lines.append("• SIGGRAPH17 shows better reconstruction quality (higher PSNR)")
            
            if eccv16_speed < siggraph17_speed:
                report_lines.append("• ECCV16 is faster for inference")
            else:
                report_lines.append("• SIGGRAPH17 is faster for inference")
            
            report_lines.append("• Both models show good performance on ImageNet dataset")
            report_lines.append("• Consider the trade-off between quality and speed for your use case")
        
        # Save report
        report_text = '\n'.join(report_lines)
        with open('imagenet_evaluation_report.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Detailed report saved as 'imagenet_evaluation_report.txt'")
        
        # Print summary to console
        print("\n" + report_text)

def main():
    print("ImageNet Evaluation Analysis")
    print("="*40)
    
    # Initialize analyzer
    analyzer = ImageNetAnalyzer()
    
    if analyzer.results:
        # Run analysis
        analyzer.analyze_metrics()
        analyzer.create_metrics_visualization()
        analyzer.create_detailed_report()
        
        print("\n🎉 Analysis complete!")
        print("Check the following files:")
        print("- imagenet_metrics_analysis.png")
        print("- imagenet_evaluation_report.txt")
    else:
        print("❌ No results to analyze. Please run the evaluation script first.")

if __name__ == "__main__":
    main()
