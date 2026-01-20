#!/usr/bin/env python3
"""
ImageNet Training Analysis Script
Analyzes training results and creates comprehensive reports
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ImageNetTrainingAnalyzer:
    def __init__(self, training_results_file='imagenet_training_results.json', 
                 evaluation_results_file='imagenet_trained_vs_pretrained_comparison.json'):
        self.training_results_file = training_results_file
        self.evaluation_results_file = evaluation_results_file
        self.training_results = None
        self.evaluation_results = None
        self.load_results()
    
    def load_results(self):
        """Load training and evaluation results"""
        # Load training results
        if os.path.exists(self.training_results_file):
            try:
                with open(self.training_results_file, 'r') as f:
                    self.training_results = json.load(f)
                print(f"✅ Loaded training results from {self.training_results_file}")
            except Exception as e:
                print(f"❌ Error loading training results: {e}")
        else:
            print(f"❌ Training results file not found: {self.training_results_file}")
        
        # Load evaluation results
        if os.path.exists(self.evaluation_results_file):
            try:
                with open(self.evaluation_results_file, 'r') as f:
                    self.evaluation_results = json.load(f)
                print(f"✅ Loaded evaluation results from {self.evaluation_results_file}")
            except Exception as e:
                print(f"❌ Error loading evaluation results: {e}")
        else:
            print(f"❌ Evaluation results file not found: {self.evaluation_results_file}")
    
    def analyze_training_progress(self):
        """Analyze training progress and convergence"""
        if not self.training_results:
            print("❌ No training results to analyze")
            return
        
        print("\n" + "="*80)
        print("IMAGENET TRAINING ANALYSIS")
        print("="*80)
        
        training_history = self.training_results.get('training_history', {})
        
        for model_name in ['eccv16', 'siggraph17']:
            if model_name in training_history:
                history = training_history[model_name]
                
                print(f"\n📊 {model_name.upper()} TRAINING ANALYSIS")
                print("-" * 50)
                
                if 'train_loss' in history and 'val_loss' in history:
                    train_losses = history['train_loss']
                    val_losses = history['val_loss']
                    epochs = history['epochs']
                    
                    # Training statistics
                    final_train_loss = train_losses[-1] if train_losses else 0
                    final_val_loss = val_losses[-1] if val_losses else 0
                    best_val_loss = min(val_losses) if val_losses else 0
                    best_epoch = epochs[val_losses.index(best_val_loss)] if val_losses else 0
                    
                    print(f"Total Epochs: {len(epochs)}")
                    print(f"Final Train Loss: {final_train_loss:.6f}")
                    print(f"Final Val Loss: {final_val_loss:.6f}")
                    print(f"Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
                    
                    # Convergence analysis
                    if len(val_losses) > 10:
                        recent_val_losses = val_losses[-10:]
                        val_loss_std = np.std(recent_val_losses)
                        val_loss_trend = np.polyfit(range(len(recent_val_losses)), recent_val_losses, 1)[0]
                        
                        print(f"Recent Val Loss Std: {val_loss_std:.6f}")
                        print(f"Val Loss Trend: {'Improving' if val_loss_trend < 0 else 'Stable' if abs(val_loss_trend) < 0.001 else 'Degrading'}")
                    
                    # Overfitting analysis
                    if len(train_losses) > 0 and len(val_losses) > 0:
                        overfitting_gap = final_val_loss - final_train_loss
                        print(f"Overfitting Gap: {overfitting_gap:.6f}")
                        print(f"Overfitting Status: {'Overfitting' if overfitting_gap > 0.01 else 'Good Generalization'}")
    
    def analyze_performance_comparison(self):
        """Analyze performance comparison between trained and pretrained models"""
        if not self.evaluation_results:
            print("❌ No evaluation results to analyze")
            return
        
        print("\n📈 PERFORMANCE COMPARISON ANALYSIS")
        print("-" * 60)
        
        results = self.evaluation_results.get('results', {})
        
        for model_name in ['eccv16', 'siggraph17']:
            pretrained_key = f"{model_name}_pretrained"
            trained_key = f"{model_name}_trained"
            
            if pretrained_key in results and trained_key in results:
                pretrained_summary = results[pretrained_key]['summary']
                trained_summary = results[trained_key]['summary']
                
                print(f"\n{model_name.upper()} MODEL COMPARISON:")
                print("-" * 40)
                
                if 'psnr' in pretrained_summary and 'psnr' in trained_summary:
                    # PSNR comparison
                    pretrained_psnr = pretrained_summary['psnr']['mean']
                    trained_psnr = trained_summary['psnr']['mean']
                    psnr_improvement = trained_psnr - pretrained_psnr
                    psnr_improvement_pct = (psnr_improvement / pretrained_psnr) * 100
                    
                    print(f"PSNR: {pretrained_psnr:.3f} → {trained_psnr:.3f} ({psnr_improvement:+.3f}, {psnr_improvement_pct:+.1f}%)")
                    
                    # SSIM comparison
                    pretrained_ssim = pretrained_summary['ssim']['mean']
                    trained_ssim = trained_summary['ssim']['mean']
                    ssim_improvement = trained_ssim - pretrained_ssim
                    ssim_improvement_pct = (ssim_improvement / pretrained_ssim) * 100
                    
                    print(f"SSIM: {pretrained_ssim:.3f} → {trained_ssim:.3f} ({ssim_improvement:+.3f}, {ssim_improvement_pct:+.1f}%)")
                    
                    # Color difference comparison
                    pretrained_color_diff = pretrained_summary['color_diff']['mean']
                    trained_color_diff = trained_summary['color_diff']['mean']
                    color_diff_improvement = pretrained_color_diff - trained_color_diff
                    color_diff_improvement_pct = (color_diff_improvement / pretrained_color_diff) * 100
                    
                    print(f"Color Diff: {pretrained_color_diff:.3f} → {trained_color_diff:.3f} ({color_diff_improvement:+.3f}, {color_diff_improvement_pct:+.1f}%)")
                    
                    # Speed comparison
                    pretrained_speed = pretrained_summary['inference_times']['mean']
                    trained_speed = trained_summary['inference_times']['mean']
                    speed_change = trained_speed - pretrained_speed
                    speed_change_pct = (speed_change / pretrained_speed) * 100
                    
                    print(f"Speed: {pretrained_speed:.2f}ms → {trained_speed:.2f}ms ({speed_change:+.2f}ms, {speed_change_pct:+.1f}%)")
                    
                    # Overall assessment
                    improvements = 0
                    if psnr_improvement > 0:
                        improvements += 1
                    if ssim_improvement > 0:
                        improvements += 1
                    if color_diff_improvement > 0:
                        improvements += 1
                    
                    print(f"Overall: {improvements}/3 metrics improved")
    
    def create_training_visualization(self):
        """Create training progress visualization"""
        if not self.training_results:
            print("❌ No training results to visualize")
            return
        
        print("\n🎨 Creating training visualization...")
        
        training_history = self.training_results.get('training_history', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ImageNet Training Progress Analysis', fontsize=16, fontweight='bold')
        
        # Plot training curves for both models
        for i, model_name in enumerate(['eccv16', 'siggraph17']):
            if model_name in training_history:
                history = training_history[model_name]
                
                if 'train_loss' in history and 'val_loss' in history:
                    epochs = history['epochs']
                    train_losses = history['train_loss']
                    val_losses = history['val_loss']
                    
                    # Training curves
                    axes[i, 0].plot(epochs, train_losses, label='Train Loss', color='blue', alpha=0.7)
                    axes[i, 0].plot(epochs, val_losses, label='Val Loss', color='red', alpha=0.7)
                    axes[i, 0].set_xlabel('Epoch')
                    axes[i, 0].set_ylabel('Loss')
                    axes[i, 0].set_title(f'{model_name.upper()} Training Curves')
                    axes[i, 0].legend()
                    axes[i, 0].grid(True, alpha=0.3)
                    
                    # Validation loss only
                    axes[i, 1].plot(epochs, val_losses, label='Val Loss', color='red', linewidth=2)
                    axes[i, 1].set_xlabel('Epoch')
                    axes[i, 1].set_ylabel('Validation Loss')
                    axes[i, 1].set_title(f'{model_name.upper()} Validation Loss')
                    axes[i, 1].legend()
                    axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('imagenet_training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Training visualization saved: imagenet_training_analysis.png")
    
    def create_performance_comparison_chart(self):
        """Create performance comparison charts"""
        if not self.evaluation_results:
            print("❌ No evaluation results to visualize")
            return
        
        print("\n📊 Creating performance comparison charts...")
        
        results = self.evaluation_results.get('results', {})
        
        # Prepare data for plotting
        models = []
        metrics = ['psnr', 'ssim', 'color_diff']
        metric_names = ['PSNR', 'SSIM', 'Color Difference']
        
        pretrained_values = {metric: [] for metric in metrics}
        trained_values = {metric: [] for metric in metrics}
        
        for model_name in ['eccv16', 'siggraph17']:
            pretrained_key = f"{model_name}_pretrained"
            trained_key = f"{model_name}_trained"
            
            if pretrained_key in results and trained_key in results:
                models.append(model_name.upper())
                
                pretrained_summary = results[pretrained_key]['summary']
                trained_summary = results[trained_key]['summary']
                
                for metric in metrics:
                    if metric in pretrained_summary and metric in trained_summary:
                        pretrained_values[metric].append(pretrained_summary[metric]['mean'])
                        trained_values[metric].append(trained_summary[metric]['mean'])
        
        # Create comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ImageNet Training Performance Comparison', fontsize=16, fontweight='bold')
        
        x = np.arange(len(models))
        width = 0.35
        
        # PSNR comparison
        axes[0, 0].bar(x - width/2, pretrained_values['psnr'], width, label='Pretrained', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, trained_values['psnr'], width, label='Trained', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('PSNR')
        axes[0, 0].set_title('PSNR Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # SSIM comparison
        axes[0, 1].bar(x - width/2, pretrained_values['ssim'], width, label='Pretrained', alpha=0.8, color='skyblue')
        axes[0, 1].bar(x + width/2, trained_values['ssim'], width, label='Trained', alpha=0.8, color='lightcoral')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].set_title('SSIM Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Color difference comparison (lower is better)
        axes[1, 0].bar(x - width/2, pretrained_values['color_diff'], width, label='Pretrained', alpha=0.8, color='skyblue')
        axes[1, 0].bar(x + width/2, trained_values['color_diff'], width, label='Trained', alpha=0.8, color='lightcoral')
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Color Difference')
        axes[1, 0].set_title('Color Difference Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Improvement summary
        improvements = []
        for i, model_name in enumerate(['eccv16', 'siggraph17']):
            psnr_improvement = trained_values['psnr'][i] - pretrained_values['psnr'][i]
            ssim_improvement = trained_values['ssim'][i] - pretrained_values['ssim'][i]
            color_diff_improvement = pretrained_values['color_diff'][i] - trained_values['color_diff'][i]
            
            total_improvement = psnr_improvement + ssim_improvement + color_diff_improvement
            improvements.append(total_improvement)
        
        axes[1, 1].bar(models, improvements, alpha=0.8, color='lightgreen')
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Total Improvement')
        axes[1, 1].set_title('Overall Improvement Summary')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('imagenet_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Performance comparison chart saved: imagenet_performance_comparison.png")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n📄 Creating comprehensive report...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("IMAGENET TRAINING COMPREHENSIVE ANALYSIS REPORT")
        report_lines.append("="*80)
        
        # Metadata
        if self.training_results:
            metadata = self.training_results.get('metadata', {})
            report_lines.append(f"\nTraining Date: {metadata.get('training_date', 'Unknown')}")
            report_lines.append(f"Device: {metadata.get('device', 'Unknown')}")
            report_lines.append(f"Dataset: {metadata.get('dataset', 'ImageNet')}")
        
        if self.evaluation_results:
            eval_metadata = self.evaluation_results.get('metadata', {})
            report_lines.append(f"Evaluation Date: {eval_metadata.get('evaluation_date', 'Unknown')}")
        
        # Training Analysis
        if self.training_results:
            report_lines.append(f"\n{'-'*60}")
            report_lines.append("TRAINING ANALYSIS")
            report_lines.append(f"{'-'*60}")
            
            training_history = self.training_results.get('training_history', {})
            
            for model_name in ['eccv16', 'siggraph17']:
                if model_name in training_history:
                    history = training_history[model_name]
                    
                    report_lines.append(f"\n{model_name.upper()} Training Results:")
                    
                    if 'train_loss' in history and 'val_loss' in history:
                        train_losses = history['train_loss']
                        val_losses = history['val_loss']
                        epochs = history['epochs']
                        
                        final_train_loss = train_losses[-1] if train_losses else 0
                        final_val_loss = val_losses[-1] if val_losses else 0
                        best_val_loss = min(val_losses) if val_losses else 0
                        best_epoch = epochs[val_losses.index(best_val_loss)] if val_losses else 0
                        
                        report_lines.append(f"  Total Epochs: {len(epochs)}")
                        report_lines.append(f"  Final Train Loss: {final_train_loss:.6f}")
                        report_lines.append(f"  Final Val Loss: {final_val_loss:.6f}")
                        report_lines.append(f"  Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
                        
                        # Convergence analysis
                        if len(val_losses) > 10:
                            recent_val_losses = val_losses[-10:]
                            val_loss_std = np.std(recent_val_losses)
                            val_loss_trend = np.polyfit(range(len(recent_val_losses)), recent_val_losses, 1)[0]
                            
                            report_lines.append(f"  Recent Val Loss Std: {val_loss_std:.6f}")
                            report_lines.append(f"  Val Loss Trend: {'Improving' if val_loss_trend < 0 else 'Stable' if abs(val_loss_trend) < 0.001 else 'Degrading'}")
        
        # Performance Comparison
        if self.evaluation_results:
            report_lines.append(f"\n{'-'*60}")
            report_lines.append("PERFORMANCE COMPARISON")
            report_lines.append(f"{'-'*60}")
            
            results = self.evaluation_results.get('results', {})
            
            for model_name in ['eccv16', 'siggraph17']:
                pretrained_key = f"{model_name}_pretrained"
                trained_key = f"{model_name}_trained"
                
                if pretrained_key in results and trained_key in results:
                    pretrained_summary = results[pretrained_key]['summary']
                    trained_summary = results[trained_key]['summary']
                    
                    report_lines.append(f"\n{model_name.upper()} Model Comparison:")
                    
                    if 'psnr' in pretrained_summary and 'psnr' in trained_summary:
                        # PSNR comparison
                        pretrained_psnr = pretrained_summary['psnr']['mean']
                        trained_psnr = trained_summary['psnr']['mean']
                        psnr_improvement = trained_psnr - pretrained_psnr
                        psnr_improvement_pct = (psnr_improvement / pretrained_psnr) * 100
                        
                        report_lines.append(f"  PSNR: {pretrained_psnr:.3f} → {trained_psnr:.3f} ({psnr_improvement:+.3f}, {psnr_improvement_pct:+.1f}%)")
                        
                        # SSIM comparison
                        pretrained_ssim = pretrained_summary['ssim']['mean']
                        trained_ssim = trained_summary['ssim']['mean']
                        ssim_improvement = trained_ssim - pretrained_ssim
                        ssim_improvement_pct = (ssim_improvement / pretrained_ssim) * 100
                        
                        report_lines.append(f"  SSIM: {pretrained_ssim:.3f} → {trained_ssim:.3f} ({ssim_improvement:+.3f}, {ssim_improvement_pct:+.1f}%)")
                        
                        # Color difference comparison
                        pretrained_color_diff = pretrained_summary['color_diff']['mean']
                        trained_color_diff = trained_summary['color_diff']['mean']
                        color_diff_improvement = pretrained_color_diff - trained_color_diff
                        color_diff_improvement_pct = (color_diff_improvement / pretrained_color_diff) * 100
                        
                        report_lines.append(f"  Color Diff: {pretrained_color_diff:.3f} → {trained_color_diff:.3f} ({color_diff_improvement:+.3f}, {color_diff_improvement_pct:+.1f}%)")
        
        # Recommendations
        report_lines.append(f"\n{'-'*60}")
        report_lines.append("RECOMMENDATIONS")
        report_lines.append(f"{'-'*60}")
        
        if self.evaluation_results:
            results = self.evaluation_results.get('results', {})
            
            for model_name in ['eccv16', 'siggraph17']:
                pretrained_key = f"{model_name}_pretrained"
                trained_key = f"{model_name}_trained"
                
                if pretrained_key in results and trained_key in results:
                    pretrained_summary = results[pretrained_key]['summary']
                    trained_summary = results[trained_key]['summary']
                    
                    if 'psnr' in pretrained_summary and 'psnr' in trained_summary:
                        psnr_improvement = trained_summary['psnr']['mean'] - pretrained_summary['psnr']['mean']
                        ssim_improvement = trained_summary['ssim']['mean'] - pretrained_summary['ssim']['mean']
                        color_diff_improvement = pretrained_summary['color_diff']['mean'] - trained_summary['color_diff']['mean']
                        
                        improvements = 0
                        if psnr_improvement > 0:
                            improvements += 1
                        if ssim_improvement > 0:
                            improvements += 1
                        if color_diff_improvement > 0:
                            improvements += 1
                        
                        if improvements >= 2:
                            report_lines.append(f"• {model_name.upper()}: Training on ImageNet shows significant improvement")
                        elif improvements == 1:
                            report_lines.append(f"• {model_name.upper()}: Training on ImageNet shows moderate improvement")
                        else:
                            report_lines.append(f"• {model_name.upper()}: Training on ImageNet shows limited improvement")
        
        report_lines.append("• Consider longer training for better convergence")
        report_lines.append("• Experiment with different learning rates and schedules")
        report_lines.append("• Use data augmentation to improve generalization")
        
        # Save report
        report_text = '\n'.join(report_lines)
        with open('imagenet_training_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Comprehensive report saved: imagenet_training_analysis_report.txt")
        
        # Print summary to console
        print("\n" + report_text)

def main():
    print("ImageNet Training Analysis")
    print("="*40)
    
    # Initialize analyzer
    analyzer = ImageNetTrainingAnalyzer()
    
    if analyzer.training_results or analyzer.evaluation_results:
        # Run analysis
        analyzer.analyze_training_progress()
        analyzer.analyze_performance_comparison()
        analyzer.create_training_visualization()
        analyzer.create_performance_comparison_chart()
        analyzer.generate_comprehensive_report()
        
        print("\n🎉 Analysis complete!")
        print("Check the following files:")
        print("- imagenet_training_analysis.png")
        print("- imagenet_performance_comparison.png")
        print("- imagenet_training_analysis_report.txt")
    else:
        print("❌ No results to analyze. Please run training and evaluation scripts first.")

if __name__ == "__main__":
    main()
