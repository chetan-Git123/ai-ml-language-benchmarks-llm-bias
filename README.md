# Reproducibility Materials for "Programming Language Adoption for AI/ML in the Era of LLMs: Empirical Benchmarks, Bias Quantification, and a Predictive Model."

This repository contains code and data to reproduce the key empirical results from the paper:

- Scaled benchmarks for matrix multiplication (up to 2000x2000) and ResNet-50 inference.
- LLM language bias study (200 prompts, standard vs. debiased conditions).
- Quantitative adoption model (simple regression).

## Requirements
- Python 3.10+
- numpy, torch, pandas, scipy, statsmodels

Install: `pip install numpy torch pandas scipy statsmodels`

## Benchmarks
Run `python benchmarks/matrix_mult.py` and `python benchmarks/resnet_inference.py` for timings (results vary by hardware; paper reports averages from 10 runs).

## LLM Bias Analysis
Data in `llm_bias/llm_bias_data.csv` (simulated based on paper results).
Run `python llm_bias/analysis.py` for chi-square test, debiased comparison, and regression model.

Results should match paper claims (e.g., 92% Python standard, p<0.001, R²~0.68 for adoption model).

License: MIT

Author: Chetan Mukhopadhyay (Independent Researcher)
