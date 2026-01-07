import pandas as pd
import numpy as np

np.random.seed(42)
n_prompts = 200

# Standard condition
langs_standard = np.random.choice(['Python', 'C++', 'Java', 'Julia', 'Rust', 'Other'],
                                  size=n_prompts,
                                  p=[0.92, 0.04, 0.02, 0.01, 0.01, 0.00])
data_standard = pd.DataFrame({'condition': 'standard', 'prompt_id': range(1, n_prompts+1), 'language': langs_standard})

# Debiased
langs_debiased = np.random.choice(['Python', 'C++', 'Java', 'Julia', 'Rust', 'Other'],
                                  size=n_prompts,
                                  p=[0.65, 0.10, 0.05, 0.08, 0.07, 0.05])
data_debiased = pd.DataFrame({'condition': 'debiased', 'prompt_id': range(1, n_prompts+1), 'language': langs_debiased})

df = pd.concat([data_standard, data_debiased])
df.to_csv('llm_bias_data.csv', index=False)
print("CSV generated with paper-approximating distributions.")
