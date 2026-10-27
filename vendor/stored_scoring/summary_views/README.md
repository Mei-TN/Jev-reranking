# Supporting data: paired contrasts, sensitivity, and training duration

These CSV files contain paired contrasts, source-weighting sensitivity checks, and training-duration comparisons computed from the recorded task summaries. Values retain their original precision.

The tables are computed from task summaries. Recomputing the underlying task scores requires the separate scoring supplement described in [DATA.md](../../../docs/DATA.md).

| File | Coverage |
| --- | --- |
| `appendix_paired_contrasts.csv` | 23 contrasts × 27 views = 621 rows. Every row has all four settings, mean, descriptive SE, positive/zero/negative counts, and extrema. Includes JevBench, original Kev six-subset, original MC11, seven-source MC, Kev test/development views, all seven individual sources, and both LOSO rules. |
| `appendix_contrast_coefficients.csv` | Explicit model-arm linear coefficients defining each contrast. |
| `appendix_loso_complete.csv` | 23 × 2 weighting rules × 7 omissions = 322 rows, containing 1,288 setting values. `omitted_source` and `loso_weighting` are separate fields. |
| `appendix_source_mapping.csv` | All eleven configurations and their seven sources. ARC Easy/Challenge have different populations; GSM8K, GPQA, and MMLU configuration pairs share their respective underlying questions. |
| `appendix_cohort_models.csv` | All 48 model rows in the four specified comparison blocks (12 reference-head models, 12 lower-rate models, 16 readout/scope models, and 8 shared-initialization scope models). This row count is separate from the 48-model E123/E458 training union, of which only 16 are selected here. Each row includes full ID, setting, hardware, rates, original endpoint values, initialization rule, and reuse relationship. |
| `appendix_budget_task_means.csv` | Eleven configuration means: both readouts at both budgets, both gaps, absolute budget changes, interaction, and all contributions divided by eleven. |
| `appendix_budget_task_settings.csv` | Five budget/readout contrasts × eleven configurations × four settings = 220 rows, each retaining the task-level effect and its original MC11 contribution. |
| `appendix_budget_absolute_settings.csv` | Eleven configurations × four settings = 44 rows, each containing pointer and pooling accuracies at both budgets. |

Setting order is F2/s44, F2/s45, F3/s44, F3/s45. Effects are calculated within each setting before averaging. Descriptive SE is sample SD divided by two. Sign counts use exact recorded values, not display rounding. LOSO ranges are ranges of seven omission-specific cohort means, not confidence intervals.

The primary JevBench family macro, Kev six-subset macro, and eleven-configuration MC11 macro remain unchanged. Seven-source weighting and Kev test-only are sensitivity views of the same retained populations. These tables are computed without additional inference or training.

Model accounting: E123 has 28 unique trainings, E458 adds 20 and reuses 4 A references, and E11 adds 8 two-epoch trainings. Thus E123/E458 has 48 unique trainings; inclusion of E11 gives 56. The E11 one-epoch pointer and pooling references reuse 8 of the original 48 models.
