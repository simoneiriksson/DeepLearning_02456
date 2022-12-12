# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:51:44 2022

@author: JUCA
"""

def treatment_profile_for_week(nm, week, treatment):
  df_cells_week_treatment = nm[(nm['week'] == week ) & (nm['Treatment'] == treatment)]
  mean_over_treatment_well_unique = df_cells_week_treatment.groupby(['Treatment', 'Image_Metadata_Compound', 'Image_Metadata_Concentration','Well_unique', 'week', 'moa'], as_index=False).mean()
  cells_week_treatment_profile = mean_over_treatment_well_unique.groupby(['Treatment', 'Image_Metadata_Compound', 'Image_Metadata_Concentration', 'week', 'moa'], as_index=False).median()
  return cells_week_treatment_profile

def treatment_profile_remaining(nm, week, treatment):
  df_cells_not_week_treatment = nm[(nm['week'] != week ) & (nm['Treatment'] != treatment)]
  mean_over_treatment_well_unique = df_cells_not_week_treatment.groupby(['Treatment', 'Image_Metadata_Compound', 'Image_Metadata_Concentration','Well_unique', 'week', 'moa'], as_index=False).mean()
  cells_not_week_treatment_profile = mean_over_treatment_well_unique.groupby(['Treatment', 'Image_Metadata_Compound', 'Image_Metadata_Concentration', 'week', 'moa'], as_index=False).median()
  return cells_not_week_treatment_profile

NSC_var = 'Image_Metadata_Compound'
class_var = 'moa'
p=2

metadata_latent_interrim = metadata_latent.groupby(['Treatment', 'week'], as_index=False).size()

treatment_week_profiles_df = pd.DataFrame(columns=metadata_latent.columns)

for index, row in metadata_latent_interrim.iterrows():
#for week, treatment in metadata_latent_interrim():
    treatment = row['Treatment']
    week = row['week']
    A_treatment_week = treatment_profile_for_week(metadata_latent, week, treatment)
    B_remain = treatment_profile_remaining(metadata_latent, week, treatment)
    
    diffs = (abs(B_remain[latent_cols] - A_treatment_week[latent_cols]))**p
    diffs_sum = diffs.sum(axis=1)**(1/p)
    diffs_min = diffs_sum.min()
    treatment_profiles_df.loc[treatment_profiles_df['Treatment']==A_treatment,'moa_pred'] = B_set.at[diffs[diffs_sum == diffs_min].index[0], 'moa']
    res = B_remain.at[diffs[diffs_sum == diffs_min].index[0], 'moa']
    A_treatment_week['moa_pred'] = res
    treatment_week_profiles_df = pd.concat([treatment_week_profiles_df, A_treatment_week], axis=0)

display(treatment_week_profiles_df)

print(treatment_week_profiles_df.groupby(['moa_pred', 'moa'])['moa_pred', 'moa'].count())

treatment_week_profiles_df['all']=1
treatment_week_profiles_df.pivot_table(values = "all", index='moa', columns='moa_pred', aggfunc=np.sum)