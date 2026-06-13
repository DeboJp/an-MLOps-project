# FinQA Benchmark Evaluation Report

*Generated on:* `2026-06-13 00:58:13`

## Evaluation Configurations
* Standard Limits: `20` examples
* RGB Testbed Limits: `50` examples

## Benchmark Results Summary

| Evaluation Set | Model Type | Examples | Program Acc | Execution Acc | Structure Error | Avg TPS | Avg TTFT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gold Chunks | Baseline | 20 | 10.0% | 20.0% | 15.0% | 56.76 | 0.434s |
| Gold Chunks | Fine-Tuned | 20 | 40.0% | 45.0% | 15.0% | 60.62 | 0.752s |
| Gold Chunks | SageMaker | 20 | 5.0% | 5.0% | 65.0% | 17.24 | 0.110s |
| Retrieved | Baseline | 20 | 0.0% | 5.0% | 30.0% | 52.16 | 0.730s |
| Retrieved | Fine-Tuned | 20 | 10.0% | 15.0% | 20.0% | 62.31 | 0.673s |
| Retrieved | SageMaker | 20 | 0.0% | 0.0% | 75.0% | 18.51 | 0.103s |
| Noise Robustness | Baseline | 50 | 8.0% | 8.0% | 16.0% | 53.79 | 0.717s |
| Noise Robustness | Fine-Tuned | 50 | 18.0% | 20.0% | 8.0% | 60.39 | 0.943s |
| Noise Robustness | SageMaker | 50 | 2.0% | 2.0% | 54.0% | 16.87 | 0.107s |
| Negative Rejection | Baseline | 50 | N/A | 20.0% (Refusal) | 28.0% | 57.76 | 0.621s |
| Negative Rejection | Fine-Tuned | 50 | N/A | 2.0% (Refusal) | 26.0% | 60.93 | 0.561s |
| Negative Rejection | SageMaker | 50 | N/A | 48.0% (Refusal) | 26.0% | 11.92 | 0.068s |
| Information Integration | Baseline | 50 | 4.0% | 6.0% | 22.0% | 53.31 | 0.690s |
| Information Integration | Fine-Tuned | 50 | 16.0% | 20.0% | 6.0% | 62.55 | 0.567s |
| Information Integration | SageMaker | 50 | 0.0% | 0.0% | 78.0% | 21.03 | 0.127s |
| Counterfactual Robustness | Baseline | 50 | 6.0% | 10.0% | 20.0% | 54.73 | 0.659s |
| Counterfactual Robustness | Fine-Tuned | 50 | 20.0% | 24.0% | 10.0% | 60.18 | 0.595s |
| Counterfactual Robustness | SageMaker | 50 | 2.0% | 2.0% | 54.0% | 19.09 | 0.115s |
