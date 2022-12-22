import pandas as pd
import numpy as np

# extracting latent variables for each image/cell
def LatentVariableExtraction(metadata, images, batch_size, vae):
    metadata['Well_unique'] = metadata['Image_Metadata_Well_DAPI'] + '_' + metadata['Image_Metadata_Plate_DAPI']
    metadata['Treatment'] = metadata['Image_Metadata_Compound'] + '_' + metadata['Image_Metadata_Concentration'].astype(str)
    metadata['week'] = metadata['Image_PathName_DAPI'].str.split("_", n=1, expand = True)[0]
    metadata['row_id'] = np.arange(len(metadata))
    images.shape[0]
    batch_size=min(batch_size, len(images))
    batch_offset = np.arange(start=0, stop=images.shape[0]+1, step=batch_size)

    df = pd.DataFrame()
    new_metadata = pd.DataFrame()

    for j, item in enumerate(batch_offset[:-1]):
        start = batch_offset[j]
        end = batch_offset[j+1]
        outputs = vae(images[start:end,:,:,:])
        z = outputs["z"]
        columns_list = ["latent_"+str(z) for z in range(z.shape[1])]
        z_df = pd.DataFrame(z.detach().numpy(), columns=columns_list)
        z_df.index = list(range(start,end))
        df = pd.concat([metadata.iloc[start:end], z_df], axis=1)
        new_metadata = pd.concat([new_metadata, df], axis=0)
        print("Profiling {}/{} batches of size {}".format(j, len(batch_offset)-1, batch_size))
    
    # last batch
    start = batch_offset[-1]
    end = images.shape[0]
    if start != end:
        #print(start, end)
        outputs = vae(images[start:end,:,:,:])
        z = outputs["z"]
        #print("z.shape", z.shape)
        columns_list = ["latent_"+str(z) for z in range(z.shape[1])]
        z_df = pd.DataFrame(z.detach().numpy(), columns=columns_list)
        z_df.index = list(range(start,end))
        df = pd.concat([metadata.iloc[start:end], z_df], axis=1)
        new_metadata = pd.concat([new_metadata, df], axis=0)
        print("Profiling {}/{} batches of size {}".format(len(batch_offset)-1, len(batch_offset)-1, end-start))

    return new_metadata

  # Wells Profiles
def well_profiles(nm):
    latent_cols = [col for col in nm.columns if type(col)==str and col[0:7]=='latent_']
    wa = nm.groupby('Image_Metadata_Well_DAPI').mean()[latent_cols]
    return wa

# function to get the cell closest to each Well profile
def well_center_cells(df,well_profiles,p=2):
    wcc = []
    latent_cols = [col for col in df.columns if type(col)==str and col[0:7]=='latent_']
    for w in well_profiles.index:
        diffs = (abs(df[df['Image_Metadata_Well_DAPI'] == w][latent_cols] - well_profiles.loc[w])**p)
        diffs_sum = diffs.sum(axis=1)**(1/p)
        diffs_min = diffs_sum.min()
        wcc.append(diffs[diffs_sum == diffs_min].index[0])
    return df.loc[wcc]

# Treatment Profiles
def treatment_profiles(nm):
    latent_cols = [col for col in nm.columns if type(col)==str and col[0:7]=='latent_']
    mean_over_treatment_well_unique = nm.groupby(['Treatment', 'Image_Metadata_Compound', 'Image_Metadata_Concentration','Well_unique', 'moa'], as_index=False).mean()
    median_over_treatment = mean_over_treatment_well_unique.groupby(['Treatment', 'Image_Metadata_Compound', 'Image_Metadata_Concentration', 'moa'], as_index=False).median()
    return median_over_treatment

# function to get the cell closest to each Treatment profile
def treatment_center_cells(df,treatment_profiles,p=2):
    tcc = []
    latent_cols = [col for col in df.columns if type(col)==str and col[0:7]=='latent_']
    for t in treatment_profiles.index:
        diffs = (abs(df[df['Treatment'] == t][latent_cols] - treatment_profiles.loc[t])**p)
        diffs_sum = diffs.sum(axis=1)**(1/p)
        diffs_min = diffs_sum.min()
        tcc.append(diffs[diffs_sum == diffs_min].index[0])
    return df.loc[tcc]


# Compount/Concentration Profiles
def CC_Profile(nm):
    latent_cols = [col for col in df.columns if type(col)==str and col[0:7]=='latent_']
    cc =  nm.groupby(['Image_Metadata_Compound','Image_Metadata_Concentration']).median()[latent_cols]
    return cc

# function to get the cell closest to each Compound/Concentration profile
def cc_center_cells(df,cc_profiles,p=2):
    cc_center_cells = []
    latent_cols = [col for col in df.columns if type(col)==str and col[0:7]=='latent_']
    for cc in cc_profiles.index:
        diffs = (abs(df[(df['Image_Metadata_Compound'] == cc[0]) & (df['Image_Metadata_Concentration'] == cc[1])][latent_cols] - cc_profiles.loc[cc]))**p
        diffs_sum = diffs.sum(axis=1)**(1/p)
        diffs_min = diffs_sum.min()
        cc_center_cells.append(diffs[diffs_sum == diffs_min].index[0])
    return df.loc[cc_center_cells]