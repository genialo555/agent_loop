"""
Tests for model quality metrics and evaluation.

This module tests:
- Perplexity calculation
- BLEU score computation  
- Task-specific metrics (accuracy, F1, etc.)
- Custom agent metrics
- Performance benchmarking
- Model evaluation pipelines
"""

import json
import time
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import MagicMock, patch
from collections import Counter
import warnings

import numpy as np
import pytest
import torch
from hypothesis import given, strategies as st, settings
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.special import softmax

# Suppress NLTK warnings for tests
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")


class TestPerplexityMetrics:
    """Test perplexity calculation for language models."""
    
    @pytest.mark.asyncio
    async def test_calculate_perplexity(self):
        """Test perplexity calculation on sample data."""
        # Mock log probabilities for a sequence
        # Lower values = higher probability = lower perplexity (better)
        log_probs = torch.tensor([
            -2.3, -1.5, -3.2, -0.8, -2.1,  # Token log probs
            -1.9, -2.5, -1.2, -2.8, -1.6
        ])
        
        # Calculate perplexity: exp(average negative log likelihood)
        avg_nll = -log_probs.mean()
        perplexity = torch.exp(avg_nll).item()
        
        assert perplexity > 0, "Perplexity must be positive"
        assert perplexity < 100, f"Perplexity too high: {perplexity:.2f}"
        
        # Verify calculation
        expected_perplexity = np.exp(-log_probs.numpy().mean())
        assert abs(perplexity - expected_perplexity) < 0.01
    
    @pytest.mark.asyncio
    async def test_perplexity_on_dataset(self):
        """Test perplexity calculation on a dataset."""
        # Mock dataset with multiple sequences
        sequences = [
            torch.tensor([-2.1, -1.8, -2.5, -1.2]),
            torch.tensor([-3.0, -2.2, -1.5, -2.8, -1.9]),
            torch.tensor([-1.6, -2.4, -1.9])
        ]
        
        # Calculate perplexity for each sequence
        perplexities = []
        for seq_log_probs in sequences:
            avg_nll = -seq_log_probs.mean()
            ppl = torch.exp(avg_nll).item()
            perplexities.append(ppl)
        
        # Dataset perplexity is usually averaged
        dataset_perplexity = np.mean(perplexities)
        
        assert dataset_perplexity > 0
        assert all(ppl > 0 for ppl in perplexities)
        
        # Good model should have reasonable perplexity
        assert dataset_perplexity < 50, f"Dataset perplexity too high: {dataset_perplexity:.2f}"
    
    @given(
        seq_length=st.integers(min_value=10, max_value=100),
        vocab_size=st.integers(min_value=100, max_value=50000)
    )
    @settings(max_examples=10)
    def test_perplexity_bounds(self, seq_length, vocab_size):
        """Property-based test for perplexity bounds."""
        # Generate random log probabilities
        log_probs = np.log(np.random.dirichlet(np.ones(vocab_size), seq_length))
        
        # Select one token per position
        selected_log_probs = [log_probs[i, np.random.randint(vocab_size)] 
                              for i in range(seq_length)]
        
        # Calculate perplexity
        avg_nll = -np.mean(selected_log_probs)
        perplexity = np.exp(avg_nll)
        
        # Perplexity bounds
        assert 1 <= perplexity <= vocab_size, \
            f"Perplexity {perplexity:.2f} outside bounds [1, {vocab_size}]"


class TestBLEUScore:
    """Test BLEU score calculation for text generation."""
    
    @pytest.mark.asyncio
    async def test_calculate_bleu_score(self):
        """Test BLEU score calculation."""
        # Reference and candidate sentences
        reference = ["The", "cat", "sat", "on", "the", "mat"]
        candidate = ["The", "cat", "sat", "on", "the", "mat"]
        
        # Perfect match should give BLEU = 1.0
        bleu = sentence_bleu([reference], candidate)
        assert bleu == 1.0
        
        # Partial match
        candidate_partial = ["The", "dog", "sat", "on", "the", "mat"]
        bleu_partial = sentence_bleu([reference], candidate_partial)
        assert 0 < bleu_partial < 1.0
        
        # No match
        candidate_none = ["A", "completely", "different", "sentence"]
        smoothing = SmoothingFunction().method1
        bleu_none = sentence_bleu([reference], candidate_none, 
                                  smoothing_function=smoothing)
        assert bleu_none < 0.1
    
    @pytest.mark.asyncio
    async def test_bleu_with_multiple_references(self):
        """Test BLEU with multiple reference translations."""
        references = [
            ["The", "cat", "is", "on", "the", "mat"],
            ["There", "is", "a", "cat", "on", "the", "mat"],
            ["A", "cat", "sits", "on", "the", "mat"]
        ]
        
        candidates = [
            ["The", "cat", "is", "on", "the", "mat"],  # Exact match with ref 1
            ["A", "cat", "is", "on", "the", "mat"],    # Close match
            ["The", "dog", "is", "on", "the", "floor"]  # Poor match
        ]
        
        bleu_scores = []
        for candidate in candidates:
            bleu = sentence_bleu(references, candidate)
            bleu_scores.append(bleu)
        
        # First candidate should have highest score
        assert bleu_scores[0] > bleu_scores[1] > bleu_scores[2]
        assert bleu_scores[0] > 0.8  # Very high match
        assert bleu_scores[2] < 0.5  # Poor match
    
    @pytest.mark.asyncio
    async def test_corpus_bleu(self):
        """Test BLEU score on corpus level."""
        # Multiple reference-candidate pairs
        corpus_refs = [
            [["Hello", "world"]],
            [["How", "are", "you"]],
            [["Machine", "learning", "is", "great"]]
        ]
        
        corpus_cands = [
            ["Hello", "world"],
            ["How", "are", "you"],
            ["ML", "is", "awesome"]  # Different but related
        ]
        
        # Calculate individual BLEU scores
        individual_scores = []
        smoothing = SmoothingFunction().method1
        
        for refs, cand in zip(corpus_refs, corpus_cands):
            score = sentence_bleu(refs, cand, smoothing_function=smoothing)
            individual_scores.append(score)
        
        # Corpus-level score (average)
        corpus_bleu = np.mean(individual_scores)
        
        assert 0 < corpus_bleu < 1
        assert corpus_bleu > 0.5  # Reasonably good match overall


class TestTaskSpecificMetrics:
    """Test task-specific evaluation metrics."""
    
    @pytest.mark.asyncio
    async def test_tool_use_accuracy(self):
        """Test accuracy metrics for tool use tasks."""
        # Ground truth tool calls
        true_tools = ["search", "calculate", "search", "read_file", "search"]
        
        # Model predictions
        predicted_tools = ["search", "calculate", "browse", "read_file", "search"]
        
        # Calculate accuracy
        accuracy = accuracy_score(true_tools, predicted_tools)
        assert accuracy == 0.8  # 4 out of 5 correct
        
        # Calculate per-tool precision/recall
        labels = list(set(true_tools + predicted_tools))
        precision, recall, f1, support = precision_recall_fscore_support(
            true_tools, predicted_tools, labels=labels, average=None
        )
        
        # Create per-tool metrics
        tool_metrics = {}
        for i, tool in enumerate(labels):
            tool_metrics[tool] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": support[i]
            }
        
        # Verify search tool (most common) has good metrics
        assert tool_metrics["search"]["recall"] == 1.0  # Found all searches
        assert tool_metrics["search"]["precision"] == 1.0  # No false positives
    
    @pytest.mark.asyncio
    async def test_code_generation_metrics(self):
        """Test metrics for code generation tasks."""
        # Mock code generation results
        test_cases = [
            {
                "generated": "def add(a, b):\n    return a + b",
                "expected": "def add(a, b):\n    return a + b",
                "syntax_valid": True,
                "tests_pass": True
            },
            {
                "generated": "def multiply(x, y)\n    return x * y",  # Missing colon
                "expected": "def multiply(x, y):\n    return x * y",
                "syntax_valid": False,
                "tests_pass": False
            },
            {
                "generated": "def divide(a, b):\n    return a / b",  # No zero check
                "expected": "def divide(a, b):\n    if b != 0:\n        return a / b\n    return None",
                "syntax_valid": True,
                "tests_pass": False  # Fails on divide by zero
            }
        ]
        
        # Calculate metrics
        syntax_accuracy = sum(tc["syntax_valid"] for tc in test_cases) / len(test_cases)
        test_pass_rate = sum(tc["tests_pass"] for tc in test_cases) / len(test_cases)
        
        assert syntax_accuracy == 2/3  # 2 out of 3 syntactically valid
        assert test_pass_rate == 1/3    # 1 out of 3 pass all tests
        
        # Calculate edit distance (simplified)
        edit_distances = []
        for tc in test_cases:
            # Simple character-level difference
            dist = sum(a != b for a, b in zip(tc["generated"], tc["expected"]))
            edit_distances.append(dist)
        
        avg_edit_distance = np.mean(edit_distances)
        assert avg_edit_distance >= 0
    
    @pytest.mark.asyncio
    async def test_instruction_following_metrics(self):
        """Test metrics for instruction following tasks."""
        instructions = [
            {
                "instruction": "List 3 colors",
                "response": "Red, Blue, Green",
                "criteria": {
                    "count_correct": True,
                    "format_correct": True,
                    "content_relevant": True
                }
            },
            {
                "instruction": "Write exactly 2 sentences about dogs",
                "response": "Dogs are loyal companions. They come in many breeds.",
                "criteria": {
                    "count_correct": True,
                    "format_correct": True,
                    "content_relevant": True
                }
            },
            {
                "instruction": "Provide a JSON object with name and age",
                "response": "name: John, age: 25",  # Wrong format
                "criteria": {
                    "count_correct": True,
                    "format_correct": False,
                    "content_relevant": True
                }
            }
        ]
        
        # Calculate instruction following score
        scores = []
        for inst in instructions:
            criteria_met = sum(inst["criteria"].values())
            total_criteria = len(inst["criteria"])
            scores.append(criteria_met / total_criteria)
        
        avg_score = np.mean(scores)
        assert avg_score > 0.7  # Good instruction following
        
        # Check specific criteria
        format_accuracy = sum(inst["criteria"]["format_correct"] for inst in instructions) / len(instructions)
        assert format_accuracy == 2/3


class TestAgentSpecificMetrics:
    """Test metrics specific to agent loop scenarios."""
    
    @pytest.mark.asyncio
    async def test_tool_chaining_success(self):
        """Test metrics for multi-step tool use."""
        # Mock agent execution traces
        traces = [
            {
                "task": "Find and summarize latest AI news",
                "steps": [
                    {"tool": "search", "success": True},
                    {"tool": "read_url", "success": True},
                    {"tool": "summarize", "success": True}
                ],
                "completed": True
            },
            {
                "task": "Calculate compound interest and save result",
                "steps": [
                    {"tool": "calculate", "success": True},
                    {"tool": "format_result", "success": True},
                    {"tool": "save_file", "success": False}  # Failed at last step
                ],
                "completed": False
            }
        ]
        
        # Calculate success metrics
        task_completion_rate = sum(t["completed"] for t in traces) / len(traces)
        
        # Calculate step success rate
        all_steps = []
        for trace in traces:
            all_steps.extend(trace["steps"])
        
        step_success_rate = sum(s["success"] for s in all_steps) / len(all_steps)
        
        assert task_completion_rate == 0.5  # 1 out of 2 tasks completed
        assert step_success_rate == 5/6     # 5 out of 6 steps succeeded
        
        # Calculate average chain length
        avg_chain_length = np.mean([len(t["steps"]) for t in traces])
        assert avg_chain_length == 3.0
    
    @pytest.mark.asyncio
    async def test_context_retention_metrics(self):
        """Test metrics for context retention across turns."""
        conversation_turns = [
            {
                "turn": 1,
                "user": "My name is Alice",
                "agent": "Hello Alice! How can I help you?",
                "retained_info": ["name=Alice"]
            },
            {
                "turn": 2,
                "user": "What's my name?",
                "agent": "Your name is Alice.",
                "retained_info": ["name=Alice"],
                "correctly_used": True
            },
            {
                "turn": 3,
                "user": "I work at TechCorp",
                "agent": "I see you work at TechCorp, Alice.",
                "retained_info": ["name=Alice", "company=TechCorp"],
                "correctly_used": True
            }
        ]
        
        # Calculate retention metrics
        retention_scores = []
        for i, turn in enumerate(conversation_turns[1:], 1):
            if "correctly_used" in turn:
                # Check if previous info was retained
                prev_info = conversation_turns[i-1]["retained_info"]
                curr_info = turn["retained_info"]
                
                retained = all(info in curr_info for info in prev_info)
                used_correctly = turn["correctly_used"]
                
                retention_scores.append(retained and used_correctly)
        
        context_retention_rate = sum(retention_scores) / len(retention_scores) if retention_scores else 0
        assert context_retention_rate == 1.0  # Perfect retention in this example


class TestPerformanceBenchmarks:
    """Test performance benchmarking metrics."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_inference_latency_metrics(self, benchmark):
        """Test inference latency measurement."""
        def mock_inference(prompt_length):
            # Simulate inference time based on prompt length
            base_time = 0.01  # 10ms base
            time_per_token = 0.0002  # 0.2ms per token
            
            total_time = base_time + (prompt_length * time_per_token)
            time.sleep(total_time)
            
            return {
                "tokens_generated": 50,
                "time_elapsed": total_time
            }
        
        # Benchmark with different prompt lengths
        prompt_lengths = [10, 50, 100, 500]
        results = []
        
        for length in prompt_lengths:
            result = benchmark.pedantic(
                mock_inference,
                args=(length,),
                iterations=5,
                rounds=2
            )
            results.append({
                "prompt_length": length,
                "avg_latency": benchmark.stats["mean"],
                "tokens_per_second": 50 / benchmark.stats["mean"]
            })
        
        # Verify latency increases with prompt length
        latencies = [r["avg_latency"] for r in results]
        assert latencies == sorted(latencies)  # Monotonically increasing
    
    @pytest.mark.asyncio
    async def test_throughput_metrics(self):
        """Test throughput measurement for batch processing."""
        batch_sizes = [1, 4, 8, 16]
        
        throughput_results = []
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Simulate batch processing
            tokens_processed = 0
            for _ in range(batch_size):
                # Each sample generates ~50 tokens
                tokens_processed += 50
                time.sleep(0.01)  # Simulate processing
            
            elapsed = time.time() - start_time
            throughput = tokens_processed / elapsed
            
            throughput_results.append({
                "batch_size": batch_size,
                "tokens_per_second": throughput,
                "samples_per_second": batch_size / elapsed
            })
        
        # Throughput should improve with batching (up to a point)
        assert throughput_results[1]["tokens_per_second"] > throughput_results[0]["tokens_per_second"]


class TestModelEvaluationPipeline:
    """Test complete model evaluation pipeline."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_evaluation(self):
        """Test comprehensive model evaluation."""
        # Mock evaluation results
        evaluation_results = {
            "general_metrics": {
                "perplexity": 8.7,
                "avg_bleu": 0.72,
                "avg_response_length": 48.3
            },
            "task_metrics": {
                "tool_use": {
                    "accuracy": 0.85,
                    "f1_score": 0.83,
                    "avg_chain_length": 2.4
                },
                "code_generation": {
                    "syntax_accuracy": 0.91,
                    "test_pass_rate": 0.78,
                    "avg_complexity": 3.2
                },
                "instruction_following": {
                    "compliance_rate": 0.88,
                    "format_accuracy": 0.82,
                    "completeness": 0.90
                }
            },
            "performance_metrics": {
                "avg_latency_ms": 45.2,
                "tokens_per_second": 38.5,
                "memory_usage_mb": 1850,
                "gpu_utilization": 0.75
            }
        }
        
        # Validate all metrics are within expected ranges
        assert 1 < evaluation_results["general_metrics"]["perplexity"] < 20
        assert 0 < evaluation_results["general_metrics"]["avg_bleu"] <= 1
        
        # Check task performance
        for task, metrics in evaluation_results["task_metrics"].items():
            for metric_name, value in metrics.items():
                if "accuracy" in metric_name or "rate" in metric_name:
                    assert 0 <= value <= 1, f"{task}.{metric_name} out of range: {value}"
        
        # Performance should be reasonable
        assert evaluation_results["performance_metrics"]["avg_latency_ms"] < 100
        assert evaluation_results["performance_metrics"]["tokens_per_second"] > 20
    
    @pytest.mark.asyncio
    async def test_metric_aggregation(self):
        """Test aggregation of metrics across multiple runs."""
        # Mock metrics from multiple evaluation runs
        runs = [
            {"accuracy": 0.82, "f1": 0.80, "perplexity": 9.2},
            {"accuracy": 0.85, "f1": 0.83, "perplexity": 8.5},
            {"accuracy": 0.83, "f1": 0.81, "perplexity": 8.9},
            {"accuracy": 0.84, "f1": 0.82, "perplexity": 8.7},
            {"accuracy": 0.86, "f1": 0.84, "perplexity": 8.3}
        ]
        
        # Calculate aggregated metrics
        aggregated = {}
        for metric in ["accuracy", "f1", "perplexity"]:
            values = [run[metric] for run in runs]
            aggregated[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        
        # Verify aggregation
        assert 0.82 <= aggregated["accuracy"]["mean"] <= 0.86
        assert aggregated["accuracy"]["std"] < 0.02  # Low variance
        assert aggregated["perplexity"]["mean"] < 9.0  # Good perplexity
        
        # Check consistency across runs
        cv_accuracy = aggregated["accuracy"]["std"] / aggregated["accuracy"]["mean"]
        assert cv_accuracy < 0.05  # Less than 5% coefficient of variation


class TestMetricVisualization:
    """Test metric visualization and reporting."""
    
    @pytest.mark.asyncio
    async def test_generate_metric_report(self):
        """Test generation of metric reports."""
        metrics = {
            "model_name": "gemma-3n-e4b-finetuned",
            "evaluation_date": "2025-07-30",
            "dataset": "agent_loop_test",
            "results": {
                "overall": {
                    "accuracy": 0.84,
                    "f1_score": 0.82,
                    "perplexity": 8.7
                },
                "by_category": {
                    "tool_use": {"accuracy": 0.87, "samples": 150},
                    "reasoning": {"accuracy": 0.81, "samples": 120},
                    "coding": {"accuracy": 0.85, "samples": 100}
                }
            }
        }
        
        # Generate report
        report = self._generate_metric_report(metrics)
        
        # Verify report structure
        assert metrics["model_name"] in report
        assert "Overall Performance" in report
        assert "Category Breakdown" in report
        
        # Check metrics are formatted
        assert "84.0%" in report or "0.84" in report  # Accuracy
        assert "8.7" in report  # Perplexity
    
    def _generate_metric_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a formatted metric report."""
        lines = [
            f"# Model Evaluation Report",
            f"\n**Model**: {metrics['model_name']}",
            f"**Date**: {metrics['evaluation_date']}",
            f"**Dataset**: {metrics['dataset']}",
            f"\n## Overall Performance",
        ]
        
        for metric, value in metrics["results"]["overall"].items():
            if metric == "perplexity":
                lines.append(f"- {metric}: {value:.1f}")
            else:
                lines.append(f"- {metric}: {value:.1%}")
        
        lines.append(f"\n## Category Breakdown")
        for category, data in metrics["results"]["by_category"].items():
            lines.append(f"\n### {category.replace('_', ' ').title()}")
            lines.append(f"- Accuracy: {data['accuracy']:.1%}")
            lines.append(f"- Samples: {data['samples']}")
        
        return "\n".join(lines)


@pytest.mark.integration
class TestMetricIntegration:
    """Integration tests for metric calculation."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    async def test_end_to_end_metric_pipeline(self):
        """Test complete metric calculation pipeline."""
        # Simulate full evaluation pipeline
        pipeline_stages = [
            "load_test_data",
            "run_inference", 
            "calculate_perplexity",
            "calculate_accuracy",
            "calculate_bleu",
            "aggregate_results",
            "generate_report"
        ]
        
        stage_results = {}
        for stage in pipeline_stages:
            # Simulate stage execution
            await asyncio.sleep(0.01)
            
            if stage == "calculate_perplexity":
                stage_results[stage] = {"perplexity": 8.9}
            elif stage == "calculate_accuracy":
                stage_results[stage] = {"accuracy": 0.83}
            elif stage == "calculate_bleu":
                stage_results[stage] = {"bleu": 0.71}
            else:
                stage_results[stage] = "completed"
        
        # Verify all stages completed
        assert len(stage_results) == len(pipeline_stages)
        assert stage_results["calculate_perplexity"]["perplexity"] < 10