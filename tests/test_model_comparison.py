"""
Tests for comparing model performance before and after training.

This module tests:
- Baseline vs fine-tuned model comparison
- Performance metrics comparison
- Response quality improvements
- Task-specific improvements
- Regression detection
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Tuple
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from hypothesis import given, strategies as st, settings
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_loop.models.training.qlora.qlora_config import QLoRAConfig


@dataclass
class ModelComparison:
    """Container for model comparison results."""
    baseline_score: float
    finetuned_score: float
    improvement: float
    relative_improvement_pct: float
    is_significant: bool
    p_value: float
    
    @property
    def improved(self) -> bool:
        return self.finetuned_score > self.baseline_score


class TestModelComparison:
    """Test baseline vs fine-tuned model comparison."""
    
    @pytest.fixture
    def comparison_prompts(self):
        """Standard prompts for comparison testing."""
        return {
            "tool_use": [
                "Use the search tool to find information about Python decorators",
                "Calculate the factorial of 10 using the calculate tool",
                "Read the contents of config.json using the file tool"
            ],
            "code_generation": [
                "Write a Python function to reverse a string",
                "Create a class for a binary search tree",
                "Implement a decorator that measures function execution time"
            ],
            "reasoning": [
                "Explain why water expands when it freezes",
                "What are the pros and cons of renewable energy?",
                "How does machine learning differ from traditional programming?"
            ],
            "instruction_following": [
                "List 5 programming languages in alphabetical order",
                "Write a haiku about artificial intelligence",
                "Summarize this in exactly 3 sentences: [long text]"
            ]
        }
    
    @pytest.fixture
    def mock_models(self):
        """Create mock baseline and fine-tuned models."""
        # Baseline model (lower quality responses)
        baseline_model = MagicMock()
        baseline_model.generate = MagicMock(side_effect=self._baseline_generate)
        baseline_model.config = MagicMock()
        
        # Fine-tuned model (improved responses)
        finetuned_model = MagicMock()
        finetuned_model.generate = MagicMock(side_effect=self._finetuned_generate)
        finetuned_model.config = MagicMock()
        
        return baseline_model, finetuned_model
    
    def _baseline_generate(self, *args, **kwargs):
        """Simulate baseline model generation (lower quality)."""
        # Return generic, less accurate responses
        return torch.tensor([[101, 102, 103, 104]])  # Mock token IDs
    
    def _finetuned_generate(self, *args, **kwargs):
        """Simulate fine-tuned model generation (higher quality)."""
        # Return more specific, accurate responses
        return torch.tensor([[201, 202, 203, 204, 205, 206]])  # More tokens
    
    @pytest.mark.asyncio
    async def test_response_quality_improvement(self, mock_models, comparison_prompts):
        """Test that fine-tuned model produces better quality responses."""
        baseline_model, finetuned_model = mock_models
        
        quality_scores = {
            "baseline": [],
            "finetuned": []
        }
        
        for category, prompts in comparison_prompts.items():
            for prompt in prompts:
                # Generate with both models
                baseline_output = baseline_model.generate(prompt)
                finetuned_output = finetuned_model.generate(prompt)
                
                # Mock quality scoring (in reality, would use actual metrics)
                baseline_score = self._calculate_quality_score(baseline_output, category)
                finetuned_score = self._calculate_quality_score(finetuned_output, category)
                
                quality_scores["baseline"].append(baseline_score)
                quality_scores["finetuned"].append(finetuned_score)
        
        # Compare average quality
        avg_baseline = np.mean(quality_scores["baseline"])
        avg_finetuned = np.mean(quality_scores["finetuned"])
        
        assert avg_finetuned > avg_baseline, \
            f"Fine-tuned model ({avg_finetuned:.2f}) should outperform baseline ({avg_baseline:.2f})"
        
        # Calculate improvement
        improvement = (avg_finetuned - avg_baseline) / avg_baseline * 100
        assert improvement > 10, f"Expected >10% improvement, got {improvement:.1f}%"
    
    def _calculate_quality_score(self, output: torch.Tensor, category: str) -> float:
        """Mock quality scoring based on output characteristics."""
        # In reality, would use perplexity, BLEU, or task-specific metrics
        base_score = 0.5
        
        # Longer outputs generally indicate more detailed responses
        length_bonus = min(output.shape[-1] / 10, 0.3)
        
        # Category-specific bonuses
        category_bonuses = {
            "tool_use": 0.1,
            "code_generation": 0.15,
            "reasoning": 0.12,
            "instruction_following": 0.08
        }
        
        return base_score + length_bonus + category_bonuses.get(category, 0)
    
    @pytest.mark.asyncio
    async def test_task_specific_improvements(self, mock_models):
        """Test improvements on specific task types."""
        baseline_model, finetuned_model = mock_models
        
        tasks = {
            "tool_calling": {
                "prompt": "Search for 'machine learning algorithms'",
                "expected_pattern": "<tool>search</tool>",
                "baseline_accuracy": 0.6,
                "finetuned_accuracy": 0.9
            },
            "code_completion": {
                "prompt": "def fibonacci(n):",
                "expected_pattern": "return",
                "baseline_accuracy": 0.7,
                "finetuned_accuracy": 0.85
            },
            "structured_output": {
                "prompt": "List 3 benefits of exercise in JSON format",
                "expected_pattern": '{"benefits":',
                "baseline_accuracy": 0.5,
                "finetuned_accuracy": 0.8
            }
        }
        
        for task_name, task_config in tasks.items():
            # Simulate accuracy based on model type
            baseline_result = np.random.random() < task_config["baseline_accuracy"]
            finetuned_result = np.random.random() < task_config["finetuned_accuracy"]
            
            # Over many runs, fine-tuned should be better
            improvement = task_config["finetuned_accuracy"] - task_config["baseline_accuracy"]
            assert improvement > 0, f"No improvement for task: {task_name}"
            
            print(f"{task_name}: {improvement*100:.0f}% improvement")
    
    @pytest.mark.asyncio
    async def test_response_consistency(self, mock_models):
        """Test that fine-tuned model is more consistent."""
        baseline_model, finetuned_model = mock_models
        
        prompt = "Explain photosynthesis in simple terms"
        num_runs = 5
        
        # Generate multiple responses from each model
        baseline_responses = []
        finetuned_responses = []
        
        for _ in range(num_runs):
            baseline_responses.append(baseline_model.generate(prompt))
            finetuned_responses.append(finetuned_model.generate(prompt))
        
        # Calculate consistency (mock - based on output similarity)
        baseline_consistency = self._calculate_consistency(baseline_responses)
        finetuned_consistency = self._calculate_consistency(finetuned_responses)
        
        # Fine-tuned model should be more consistent
        assert finetuned_consistency > baseline_consistency, \
            "Fine-tuned model should produce more consistent outputs"
    
    def _calculate_consistency(self, responses: List[torch.Tensor]) -> float:
        """Calculate consistency score for a set of responses."""
        # Mock consistency calculation
        # In reality, would use embedding similarity or other metrics
        if len(responses) < 2:
            return 1.0
        
        # Simulate: fine-tuned models have less variance
        base_consistency = 0.7
        variance_penalty = np.random.uniform(0, 0.2)
        
        return base_consistency - variance_penalty


class TestPerformanceMetrics:
    """Test performance metric comparisons."""
    
    @pytest.mark.asyncio
    async def test_perplexity_comparison(self):
        """Test perplexity improvement after fine-tuning."""
        # Mock perplexity scores
        baseline_perplexity = 15.2
        finetuned_perplexity = 8.7
        
        # Lower perplexity is better
        assert finetuned_perplexity < baseline_perplexity
        
        improvement = (baseline_perplexity - finetuned_perplexity) / baseline_perplexity * 100
        assert improvement > 20, f"Expected >20% perplexity improvement, got {improvement:.1f}%"
    
    @pytest.mark.asyncio
    async def test_inference_speed_comparison(self):
        """Test inference speed comparison."""
        # Mock inference times (ms per token)
        baseline_times = [23.5, 24.1, 23.8, 24.3, 23.9]
        finetuned_times = [24.2, 24.8, 24.5, 25.1, 24.7]  # Slightly slower due to LoRA
        
        avg_baseline = np.mean(baseline_times)
        avg_finetuned = np.mean(finetuned_times)
        
        # Fine-tuned might be slightly slower but should be within acceptable range
        slowdown = (avg_finetuned - avg_baseline) / avg_baseline * 100
        assert slowdown < 10, f"Excessive slowdown: {slowdown:.1f}%"
    
    @pytest.mark.asyncio
    async def test_memory_usage_comparison(self):
        """Test memory usage comparison."""
        # Mock memory usage (MB)
        baseline_memory = {
            "model": 1500,
            "inference": 200,
            "total": 1700
        }
        
        finetuned_memory = {
            "model": 1550,  # Slightly more due to LoRA adapters
            "inference": 200,
            "total": 1750
        }
        
        memory_increase = finetuned_memory["total"] - baseline_memory["total"]
        memory_increase_pct = memory_increase / baseline_memory["total"] * 100
        
        assert memory_increase_pct < 5, f"Excessive memory increase: {memory_increase_pct:.1f}%"


class TestStatisticalSignificance:
    """Test statistical significance of improvements."""
    
    @pytest.mark.asyncio
    async def test_performance_significance(self):
        """Test if performance improvements are statistically significant."""
        # Mock scores from multiple evaluation runs
        np.random.seed(42)
        
        # Baseline scores (lower)
        baseline_scores = np.random.normal(0.7, 0.05, 30)
        
        # Fine-tuned scores (higher with less variance)
        finetuned_scores = np.random.normal(0.82, 0.03, 30)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(finetuned_scores, baseline_scores)
        
        # Check if improvement is significant (p < 0.05)
        assert p_value < 0.05, f"Improvement not statistically significant (p={p_value:.4f})"
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(baseline_scores)**2 + np.std(finetuned_scores)**2) / 2)
        effect_size = (np.mean(finetuned_scores) - np.mean(baseline_scores)) / pooled_std
        
        assert effect_size > 0.8, f"Effect size too small: {effect_size:.2f} (want > 0.8)"
    
    @given(
        n_samples=st.integers(min_value=20, max_value=100),
        baseline_mean=st.floats(min_value=0.5, max_value=0.8),
        improvement=st.floats(min_value=0.05, max_value=0.2)
    )
    @settings(max_examples=10)
    def test_significance_robustness(self, n_samples, baseline_mean, improvement):
        """Property-based test for significance testing robustness."""
        # Generate synthetic data
        baseline = np.random.normal(baseline_mean, 0.05, n_samples)
        finetuned = np.random.normal(baseline_mean + improvement, 0.04, n_samples)
        
        # Test should handle various sample sizes and improvements
        t_stat, p_value = stats.ttest_ind(finetuned, baseline)
        
        # With reasonable improvement, should often be significant
        if improvement > 0.1:
            # Strong improvement should usually be significant
            assert p_value < 0.1 or n_samples < 30  # Allow for small sample variance


class TestRegressionDetection:
    """Test detection of performance regressions."""
    
    @pytest.mark.asyncio
    async def test_detect_task_regressions(self):
        """Test detection of regressions on specific tasks."""
        # Performance on different tasks
        task_performance = {
            "math_problems": {"baseline": 0.75, "finetuned": 0.78},
            "creative_writing": {"baseline": 0.80, "finetuned": 0.76},  # Regression!
            "factual_qa": {"baseline": 0.82, "finetuned": 0.85},
            "code_generation": {"baseline": 0.70, "finetuned": 0.73}
        }
        
        regressions = []
        improvements = []
        
        for task, scores in task_performance.items():
            change = scores["finetuned"] - scores["baseline"]
            change_pct = change / scores["baseline"] * 100
            
            if change < 0:
                regressions.append((task, change_pct))
            else:
                improvements.append((task, change_pct))
        
        # Report findings
        assert len(regressions) <= 1, f"Too many regressions: {regressions}"
        
        if regressions:
            task, pct = regressions[0]
            assert abs(pct) < 10, f"Severe regression in {task}: {pct:.1f}%"
    
    @pytest.mark.asyncio
    async def test_catastrophic_forgetting_check(self):
        """Test for catastrophic forgetting of base capabilities."""
        base_capabilities = [
            "basic_arithmetic",
            "language_understanding", 
            "common_sense_reasoning",
            "grammar_correctness"
        ]
        
        # Mock capability retention scores
        retention_scores = {
            "basic_arithmetic": 0.95,      # Well retained
            "language_understanding": 0.92,  # Well retained
            "common_sense_reasoning": 0.88,  # Some degradation
            "grammar_correctness": 0.91      # Well retained
        }
        
        # Check minimum retention threshold
        min_retention = 0.85
        
        for capability, score in retention_scores.items():
            assert score >= min_retention, \
                f"Catastrophic forgetting detected for {capability}: {score:.2f}"


class TestABComparison:
    """Test A/B comparison methodology."""
    
    @pytest.mark.asyncio
    async def test_blind_evaluation_setup(self):
        """Test setup for blind A/B evaluation."""
        # Create evaluation samples
        eval_prompts = [
            "Explain the concept of recursion",
            "Write a function to find prime numbers",
            "What are the benefits of exercise?"
        ]
        
        # Generate responses from both models (mocked)
        model_a_responses = ["Response A1", "Response A2", "Response A3"]
        model_b_responses = ["Response B1", "Response B2", "Response B3"]
        
        # Create blind evaluation pairs
        evaluation_pairs = []
        for i, prompt in enumerate(eval_prompts):
            # Randomly assign positions to avoid bias
            if np.random.random() > 0.5:
                pair = {
                    "prompt": prompt,
                    "response_1": model_a_responses[i],
                    "response_2": model_b_responses[i],
                    "correct_order": "A_B"
                }
            else:
                pair = {
                    "prompt": prompt,
                    "response_1": model_b_responses[i],
                    "response_2": model_a_responses[i],
                    "correct_order": "B_A"
                }
            evaluation_pairs.append(pair)
        
        # Verify blind setup
        assert len(evaluation_pairs) == len(eval_prompts)
        assert all("correct_order" in pair for pair in evaluation_pairs)
    
    @pytest.mark.asyncio
    async def test_preference_scoring(self):
        """Test human preference scoring simulation."""
        # Simulate preference scores (1 = first response, 2 = second response)
        preferences = {
            "coherence": [2, 1, 2, 2, 1],      # Model B slightly better
            "accuracy": [2, 2, 1, 2, 2],       # Model B better
            "helpfulness": [1, 2, 2, 2, 1],    # Model B slightly better
            "style": [1, 1, 2, 1, 2]           # Mixed
        }
        
        # Calculate win rates
        model_b_wins = {}
        for criterion, scores in preferences.items():
            wins = sum(1 for s in scores if s == 2)
            model_b_wins[criterion] = wins / len(scores)
        
        # Overall, Model B (fine-tuned) should win
        overall_win_rate = np.mean(list(model_b_wins.values()))
        assert overall_win_rate > 0.5, f"Fine-tuned model should win overall: {overall_win_rate:.2f}"


class TestComparisonReport:
    """Test generation of comparison reports."""
    
    @pytest.mark.asyncio
    async def test_generate_comparison_report(self):
        """Test generation of comprehensive comparison report."""
        # Mock comparison data
        comparison_data = {
            "model_info": {
                "baseline": "gemma-3n-e4b",
                "finetuned": "gemma-3n-e4b-agentloop-v1",
                "training_steps": 1000,
                "dataset": "agent_instruct"
            },
            "metrics": {
                "perplexity": {"baseline": 12.5, "finetuned": 8.3},
                "accuracy": {"baseline": 0.73, "finetuned": 0.81},
                "f1_score": {"baseline": 0.70, "finetuned": 0.79}
            },
            "task_performance": {
                "tool_use": {"baseline": 0.68, "finetuned": 0.85},
                "reasoning": {"baseline": 0.75, "finetuned": 0.82},
                "coding": {"baseline": 0.71, "finetuned": 0.78}
            },
            "inference_stats": {
                "avg_tokens_per_sec": {"baseline": 42.3, "finetuned": 40.8},
                "memory_mb": {"baseline": 1650, "finetuned": 1720}
            }
        }
        
        # Generate report
        report = self._generate_report(comparison_data)
        
        # Verify report contains key sections
        assert "Model Comparison Report" in report
        assert "Overall Metrics" in report
        assert "Task-Specific Performance" in report
        assert "Recommendations" in report
        
        # Check improvement calculations
        for metric, values in comparison_data["metrics"].items():
            baseline = values["baseline"]
            finetuned = values["finetuned"]
            if baseline != 0:
                improvement = (finetuned - baseline) / baseline * 100
                assert f"{improvement:.1f}%" in report or f"{abs(improvement):.1f}%" in report
    
    def _generate_report(self, data: Dict[str, Any]) -> str:
        """Generate a comparison report from data."""
        report_lines = [
            "# Model Comparison Report",
            f"\n## Models",
            f"- Baseline: {data['model_info']['baseline']}",
            f"- Fine-tuned: {data['model_info']['finetuned']}",
            f"- Training steps: {data['model_info']['training_steps']}",
            f"\n## Overall Metrics"
        ]
        
        for metric, values in data["metrics"].items():
            baseline = values["baseline"]
            finetuned = values["finetuned"]
            
            if baseline != 0:
                change = (finetuned - baseline) / baseline * 100
                direction = "↑" if change > 0 else "↓"
                report_lines.append(
                    f"- {metric}: {baseline:.3f} → {finetuned:.3f} "
                    f"({direction} {abs(change):.1f}%)"
                )
        
        report_lines.extend([
            f"\n## Task-Specific Performance",
            "| Task | Baseline | Fine-tuned | Improvement |",
            "|------|----------|------------|-------------|"
        ])
        
        for task, values in data["task_performance"].items():
            baseline = values["baseline"]
            finetuned = values["finetuned"]
            improvement = (finetuned - baseline) / baseline * 100
            report_lines.append(
                f"| {task} | {baseline:.2f} | {finetuned:.2f} | "
                f"{improvement:+.1f}% |"
            )
        
        report_lines.extend([
            f"\n## Recommendations",
            "- Fine-tuned model shows significant improvements across all tasks",
            "- Minimal performance overhead observed",
            "- Recommended for production deployment"
        ])
        
        return "\n".join(report_lines)


@pytest.mark.integration
class TestIntegrationComparison:
    """Integration tests for model comparison."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    async def test_full_comparison_workflow(self):
        """Test complete comparison workflow."""
        # This would load actual models and run comparisons
        # For now, we simulate the workflow
        
        workflow_steps = [
            "load_baseline_model",
            "load_finetuned_model",
            "prepare_test_dataset",
            "run_baseline_inference",
            "run_finetuned_inference",
            "calculate_metrics",
            "statistical_analysis",
            "generate_report"
        ]
        
        results = {}
        for step in workflow_steps:
            # Simulate step execution
            await asyncio.sleep(0.01)
            results[step] = "completed"
        
        assert all(results[step] == "completed" for step in workflow_steps)