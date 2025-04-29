# %%%
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('CCRB-Complaint-Data_202007271729/allegations_202007271729.csv')
def fix_ethnicity(x):
    return 'Other' if x in ['Other Race', 'Unknown', 'Refused'] or pd.isna(x) else x
df['complainant_ethnicity'] = df['complainant_ethnicity'].apply(fix_ethnicity)
df['mos_ethnicity'] = df['mos_ethnicity'].apply(fix_ethnicity)
ethnicities = pd.concat([df['complainant_ethnicity'], df['mos_ethnicity']], axis=0).unique()
palette = sns.color_palette("husl", n_colors=len(ethnicities))
sns.set_palette(palette)

# %% [markdown]
# # Proposition: Blacks are Systematically Targeted for Police Misconduct
# %%
grouped_data = df.groupby(['mos_ethnicity', 'complainant_ethnicity']).size().unstack()

grouped_data = grouped_data.fillna(0)

normalized_data = grouped_data.div(grouped_data.sum(axis=1), axis=0)

normalized_data_reset = normalized_data.reset_index()
normalized_data_melted = normalized_data_reset.melt(id_vars='mos_ethnicity', var_name='complainant_ethnicity', value_name='normalized_proportion')

plt.figure(figsize=(12, 8))
sns.barplot(
    data=normalized_data_melted,
    x='mos_ethnicity',
    y='normalized_proportion',
    hue='complainant_ethnicity',
)
plt.title("Alleged Police Misconduct by Officer and Complainant Ethnicity")
plt.xlabel("Officer Ethnicity")
plt.ylabel("Proportion of Complaints")
plt.legend(title="Complainant Ethnicity", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('allegations_by_officer_and_complainant_ethnicity_barplot.png', dpi=300)
plt.show()

# %% [markdown]
# # Proposition: Blacks are Systematically Targeted for Police Misconduct
# %%
def sigmoid_emphasis(x, center=0.525, steepness=7.5):
    """Use sigmoid function to make the color palette more extreme"""
    return 1 / (1 + np.exp(-steepness * (x - center)))
normalized_data_sigmoid = sigmoid_emphasis(normalized_data)
plt.figure(figsize=(12, 8))
sns.heatmap(normalized_data_sigmoid.T, annot=normalized_data.T.round(2), fmt=".2f", cmap="YlOrRd", linewidths=0.5)
plt.title("Alleged Police Misconduct by Officer and Complainant Ethnicity")
plt.xlabel("Officer Ethnicity")
plt.ylabel("Complainant Ethnicity")

# Bold the "Black" label on y-axis
ytick_labels = plt.gca().get_yticklabels()
for i, label in enumerate(ytick_labels):
    if label.get_text() == "Black":
        ytick_labels[i].set_fontweight('bold')
plt.gca().set_yticklabels(ytick_labels)

plt.savefig('allegations_by_officer_and_complainant_ethnicity_heatmap.png', dpi=300)
plt.show()

# %% [markdown]
# # Counter-Proposition: Blacks are Not Systematically Targeted for Police Misconduct
# %%
queried_data = df.query('precinct == 110.0')
queried_counts = queried_data['complainant_ethnicity'].value_counts()
queried_normalized = queried_counts / queried_counts.sum()
queried_normalized_melted = queried_normalized.reset_index().rename(columns={'index': 'complainant_ethnicity', 'count': 'normalized_proportion'})
plt.figure(figsize=(12, 8))
sns.barplot(
    data=queried_normalized_melted,
    x='complainant_ethnicity',
    y='normalized_proportion',
    hue='complainant_ethnicity',
)
plt.title("Alleged Police Misconduct by Complainant Ethnicity$^{*}$")
plt.xlabel("Complainant Ethnicity")
plt.ylabel("Proportion of Complaints")
plt.text(1.0, -0.1, "$^{*}$Note: This is only for precinct 110", ha='right', va='bottom', transform=plt.gca().transAxes)
plt.savefig('allegations_by_complainant_ethnicity_barplot_110.png', dpi=300)
plt.show()

# %% [markdown]
# # Counter-Proposition: Blacks are Not Systematically Targeted for Police Misconduct
# %%
# Group by officer and complainant ethnicity
grouped_110 = queried_data.groupby(['mos_ethnicity', 'complainant_ethnicity']).size().unstack().fillna(0)

# Normalize so each row sums to 1 (proportion of complaints by complainant ethnicity for each officer ethnicity)
normalized_110 = grouped_110.div(grouped_110.sum(axis=1), axis=0)
normalized_110_sigmoid = sigmoid_emphasis(normalized_110, center=0.53, steepness=5)

plt.figure(figsize=(12, 8))
sns.heatmap(normalized_110_sigmoid.T, annot=normalized_110.T.round(2), fmt=".2f", cmap="YlOrRd", linewidths=0.5)
# plt.title("Alleged Police Misconduct by Officer and Complainant Ethnicity$^{*}$")
plt.title("Alleged Police Misconduct by Officer and Complainant Ethnicity")
plt.xlabel("Officer Ethnicity")
plt.ylabel("Complainant Ethnicity")
# plt.text(1.15, -0.1, "$^{*}$Note: This is only for precinct 110", ha='right', va='bottom', transform=plt.gca().transAxes)
plt.savefig('allegations_by_officer_and_complainant_ethnicity_heatmap_110.png', dpi=300)
plt.show()
# %% [markdown]
# # Writeup
#
# - Proposition: blacks are systematically targeted for police misconduct.
# - Deceptive Technique: I changed the color palette for the heatmap to make the difference between the values for black complainants between other ethnicities more extreme.
# - Counter-Proposition: blacks are not systematically targeted for police misconduct.
# - Deceptive Technique: I hand-picked a precinct with a low proportion of misconduct allegations against blacks to make the counter-proposition look plausible.
# - Deceptive Technique: I changed the color palette for the heatmap to make the difference between the values for non-black complainants between other ethnicities more extreme.