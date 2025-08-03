#!/usr/bin/env python3
"""
Model Drift Detection Script

Detects statistical drift in model inputs and outputs compared to reference data.
Uses statistical tests and distribution analysis to identify potential model degradation.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


class ModelDriftDetector:
    """
    Detects drift in model inputs and outputs using statistical methods.
    
    Methods:
    - Kolmogorov-Smirnov test for distribution drift
    - Population Stability Index (PSI) for feature drift
    - Cosine similarity for embedding drift
    - Jensen-Shannon divergence for probability distributions
    """
    
    def __init__(self, threshold: float = 0.1, confidence: float = 0.05):
        self.threshold = threshold
        self.confidence = confidence
        self.logger = logging.getLogger(__name__)
        
    def detect_input_drift(
        self, 
        reference_data: pd.DataFrame, 
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect drift in input features using multiple statistical tests.
        
        Args:
            reference_data: Historical reference data
            current_data: Current production data
            
        Returns:
            Dictionary with drift detection results
        """
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'drift_detected': False,
            'features': {},
            'overall_score': 0.0,
            'threshold': self.threshold
        }
        
        drift_scores = []
        
        for column in reference_data.columns:
            if column in current_data.columns:
                feature_result = self._analyze_feature_drift(
                    reference_data[column], 
                    current_data[column],
                    column
                )
                results['features'][column] = feature_result
                drift_scores.append(feature_result['drift_score'])
                
        # Calculate overall drift score
        results['overall_score'] = np.mean(drift_scores) if drift_scores else 0.0
        results['drift_detected'] = results['overall_score'] > self.threshold
        
        self.logger.info(f"Input drift analysis complete. Overall score: {results['overall_score']:.4f}")
        
        return results
        
    def _analyze_feature_drift(
        self, 
        reference: pd.Series, 
        current: pd.Series, 
        feature_name: str
    ) -> Dict[str, Any]:
        """Analyze drift for a single feature using multiple methods."""
        
        result = {
            'feature_name': feature_name,
            'drift_score': 0.0,
            'tests': {},
            'drift_detected': False
        }
        
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(reference.dropna(), current.dropna())
            result['tests']['kolmogorov_smirnov'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_pvalue),
                'drift_detected': ks_pvalue < self.confidence
            }
            
            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(reference, current)
            result['tests']['population_stability_index'] = {
                'score': float(psi_score),
                'drift_detected': psi_score > 0.1  # PSI threshold
            }
            
            # Mann-Whitney U test (non-parametric)
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                reference.dropna(), 
                current.dropna(), 
                alternative='two-sided'
            )
            result['tests']['mann_whitney'] = {
                'statistic': float(mw_stat),
                'p_value': float(mw_pvalue),
                'drift_detected': mw_pvalue < self.confidence
            }
            
            # Calculate composite drift score
            drift_indicators = [
                ks_stat,
                psi_score,
                1 - mw_pvalue  # Convert p-value to drift indicator
            ]
            result['drift_score'] = float(np.mean(drift_indicators))
            result['drift_detected'] = result['drift_score'] > self.threshold
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature {feature_name}: {e}")
            result['error'] = str(e)
            
        return result
        
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in distribution between two samples.
        PSI < 0.1: No significant shift
        0.1 <= PSI < 0.2: Moderate shift
        PSI >= 0.2: Significant shift
        """
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference.dropna(), bins=buckets)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference.dropna(), bins=bin_edges)
            cur_counts, _ = np.histogram(current.dropna(), bins=bin_edges)
            
            # Convert to percentages and add small epsilon to avoid log(0)
            epsilon = 1e-6
            ref_pct = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * buckets)
            cur_pct = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * buckets)
            
            # Calculate PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            
            return float(psi)
            
        except Exception as e:
            self.logger.warning(f"PSI calculation failed: {e}")
            return 0.0
            
    def detect_output_drift(
        self, 
        reference_outputs: List[str], 
        current_outputs: List[str]
    ) -> Dict[str, Any]:
        """
        Detect drift in model outputs using text similarity and distribution analysis.
        
        Args:
            reference_outputs: Historical model outputs
            current_outputs: Current model outputs
            
        Returns:
            Dictionary with output drift analysis
        """
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'drift_detected': False,
            'metrics': {},
            'overall_score': 0.0
        }
        
        try:
            # Length distribution analysis
            ref_lengths = [len(output) for output in reference_outputs]
            cur_lengths = [len(output) for output in current_outputs]
            
            length_ks_stat, length_ks_pvalue = stats.ks_2samp(ref_lengths, cur_lengths)
            
            result['metrics']['length_distribution'] = {
                'ks_statistic': float(length_ks_stat),
                'p_value': float(length_ks_pvalue),
                'drift_detected': length_ks_pvalue < self.confidence,
                'reference_mean_length': float(np.mean(ref_lengths)),
                'current_mean_length': float(np.mean(cur_lengths))
            }
            
            # Word frequency analysis
            ref_words = self._extract_word_frequencies(reference_outputs)
            cur_words = self._extract_word_frequencies(current_outputs)
            
            # Jensen-Shannon divergence for word distributions
            js_divergence = self._calculate_js_divergence(ref_words, cur_words)
            
            result['metrics']['word_distribution'] = {
                'js_divergence': float(js_divergence),
                'drift_detected': js_divergence > 0.1,  # JS divergence threshold
                'reference_vocab_size': len(ref_words),
                'current_vocab_size': len(cur_words)
            }
            
            # Calculate overall output drift score
            drift_scores = [
                length_ks_stat,
                js_divergence
            ]
            result['overall_score'] = float(np.mean(drift_scores))
            result['drift_detected'] = result['overall_score'] > self.threshold
            
        except Exception as e:
            self.logger.error(f"Output drift analysis failed: {e}")
            result['error'] = str(e)
            
        return result
        
    def _extract_word_frequencies(self, texts: List[str]) -> Dict[str, float]:
        """Extract normalized word frequencies from texts."""
        word_counts = {}
        total_words = 0
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                # Simple tokenization - could be improved with proper NLP
                clean_word = ''.join(c for c in word if c.isalnum())
                if clean_word:
                    word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
                    total_words += 1
                    
        # Normalize to frequencies
        return {word: count / total_words for word, count in word_counts.items()}
        
    def _calculate_js_divergence(
        self, 
        dist1: Dict[str, float], 
        dist2: Dict[str, float]
    ) -> float:
        """Calculate Jensen-Shannon divergence between two probability distributions."""
        try:
            # Get union of all words
            all_words = set(dist1.keys()) | set(dist2.keys())
            
            # Create probability vectors
            p = np.array([dist1.get(word, 1e-10) for word in all_words])
            q = np.array([dist2.get(word, 1e-10) for word in all_words])
            
            # Normalize to ensure they sum to 1
            p = p / np.sum(p)
            q = q / np.sum(q)
            
            # Calculate JS divergence
            m = (p + q) / 2
            js_div = (stats.entropy(p, m) + stats.entropy(q, m)) / 2
            
            return float(js_div)
            
        except Exception as e:
            self.logger.warning(f"JS divergence calculation failed: {e}")
            return 0.0


class ModelDriftMonitor:
    """
    Monitors model drift by collecting data from production API and comparing
    with reference datasets.
    """
    
    def __init__(self, endpoint: str, reference_data_path: str):
        self.endpoint = endpoint
        self.reference_data_path = Path(reference_data_path)
        self.detector = ModelDriftDetector()
        self.logger = logging.getLogger(__name__)
        
    async def collect_production_data(self, sample_size: int = 100) -> List[Dict[str, Any]]:
        """Collect sample data from production API."""
        samples = []
        
        async with httpx.AsyncClient() as client:
            for i in range(sample_size):
                try:
                    # Sample request - adjust based on your API
                    response = await client.post(
                        f"{self.endpoint}/run-agent",
                        json={
                            "instruction": f"Sample test instruction {i}",
                            "use_ollama": True,
                            "temperature": 0.7
                        },
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        samples.append({
                            'timestamp': datetime.utcnow().isoformat(),
                            'input': f"Sample test instruction {i}",
                            'output': result.get('result', ''),
                            'execution_time_ms': result.get('execution_time_ms', 0),
                            'success': result.get('success', False)
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Failed to collect sample {i}: {e}")
                    
        self.logger.info(f"Collected {len(samples)} production samples")
        return samples
        
    def load_reference_data(self) -> Dict[str, Any]:
        """Load reference dataset for comparison."""
        try:
            with open(self.reference_data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load reference data: {e}")
            return {}
            
    async def run_drift_analysis(self, sample_size: int = 100) -> Dict[str, Any]:
        """Run complete drift analysis."""
        self.logger.info("Starting drift analysis...")
        
        # Collect current production data
        current_samples = await self.collect_production_data(sample_size)
        
        # Load reference data
        reference_data = self.load_reference_data()
        
        if not reference_data or not current_samples:
            return {
                'error': 'Insufficient data for drift analysis',
                'reference_samples': len(reference_data.get('samples', [])),
                'current_samples': len(current_samples)
            }
            
        # Extract features for analysis
        current_df = pd.DataFrame([
            {
                'output_length': len(sample['output']),
                'execution_time_ms': sample['execution_time_ms'],
                'success': int(sample['success'])
            }
            for sample in current_samples
        ])
        
        reference_df = pd.DataFrame([
            {
                'output_length': len(sample['output']),
                'execution_time_ms': sample['execution_time_ms'],
                'success': int(sample['success'])
            }
            for sample in reference_data.get('samples', [])
        ])
        
        # Run input drift analysis
        input_drift = self.detector.detect_input_drift(reference_df, current_df)
        
        # Run output drift analysis
        current_outputs = [sample['output'] for sample in current_samples]
        reference_outputs = [sample['output'] for sample in reference_data.get('samples', [])]
        output_drift = self.detector.detect_output_drift(reference_outputs, current_outputs)
        
        # Compile final report
        report = {
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'sample_sizes': {
                'reference': len(reference_data.get('samples', [])),
                'current': len(current_samples)
            },
            'input_drift': input_drift,
            'output_drift': output_drift,
            'overall_drift_detected': input_drift['drift_detected'] or output_drift['drift_detected'],
            'recommendations': self._generate_recommendations(input_drift, output_drift)
        }
        
        return report
        
    def _generate_recommendations(
        self, 
        input_drift: Dict[str, Any], 
        output_drift: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []
        
        if input_drift['drift_detected']:
            recommendations.append(
                "Input drift detected. Consider retraining the model with recent data."
            )
            
            # Feature-specific recommendations
            for feature, result in input_drift['features'].items():
                if result['drift_detected']:
                    recommendations.append(
                        f"Feature '{feature}' shows significant drift (score: {result['drift_score']:.3f}). "
                        f"Investigate data pipeline changes."
                    )
                    
        if output_drift['drift_detected']:
            recommendations.append(
                "Output drift detected. Model responses have changed significantly. "
                "Review model performance and consider rollback."
            )
            
            if output_drift['metrics'].get('length_distribution', {}).get('drift_detected'):
                recommendations.append(
                    "Output length distribution has changed. Check for prompt engineering modifications."
                )
                
            if output_drift['metrics'].get('word_distribution', {}).get('drift_detected'):
                recommendations.append(
                    "Output vocabulary has shifted. Verify model fine-tuning and data quality."
                )
                
        if not input_drift['drift_detected'] and not output_drift['drift_detected']:
            recommendations.append(
                "No significant drift detected. Model is performing within expected parameters."
            )
            
        return recommendations


async def main():
    """Main entry point for drift detection script."""
    parser = argparse.ArgumentParser(description="Model Drift Detection")
    parser.add_argument("--endpoint", required=True, help="Production API endpoint")
    parser.add_argument("--reference-data", required=True, help="Path to reference dataset")
    parser.add_argument("--threshold", type=float, default=0.1, help="Drift detection threshold")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples to collect")
    parser.add_argument("--output", help="Output file for drift report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize monitor
        monitor = ModelDriftMonitor(args.endpoint, args.reference_data)
        monitor.detector.threshold = args.threshold
        
        # Run drift analysis
        report = await monitor.run_drift_analysis(args.sample_size)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Drift report saved to {args.output}")
        else:
            print(json.dumps(report, indent=2))
            
        # Exit with appropriate code
        if report.get('overall_drift_detected', False):
            logger.warning("⚠️ Model drift detected!")
            sys.exit(1)
        else:
            logger.info("✅ No significant drift detected")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())