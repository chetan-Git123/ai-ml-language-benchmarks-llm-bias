import pandas as pd
from scipy.stats import chisquare
import statsmodels.api as sm
import numpy as np

df = pd.read_csv('llm_bias_data.csv')

# Chi-square (standard vs. uniform null)
observed = df[df['condition']=='standard']['language'].value_counts()
expected = np.array([n_prompts / 6] * 6)  # Uniform
chi2, p = chisquare(observed, expected)
print(f"Chi-square test (standard): chi2={chi2:.2f}, p={p:.4f}")

# Debiased reduction
std_py = (df[df['condition']=='standard']['language'] == 'Python').mean()
deb_py = (df[df['condition']=='debiased']['language'] == 'Python').mean()
print(f"Python % standard: {std_py*100:.1f}%, debiased: {deb_py*100:.1f}%")

# Adoption model (simulated data from paper)
data = {'adoption': [10, 6, 3, 4, 5],  # Arbitrary: Python high, others lower
        'performance': [4, 10, 9.8, 8.7, 8.2],
        'llm_exposure': [10, 5, 2, 3, 4]}
df_model = pd.DataFrame(data)
X = df_model[['performance', 'llm_exposure']]
X = sm.add_constant(X)
y = df_model['adoption']
model = sm.OLS(y, X).fit()
print(model.summary())
