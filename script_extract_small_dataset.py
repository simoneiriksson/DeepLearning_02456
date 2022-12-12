from utils.data_preparation import *

metadata = read_metadata("./data/all/metadata.csv")
metadata = drop_redundant_metadata_columns(metadata)

metadata_subset = metadata[(metadata['Multi_Cell_Image_Id'] <= 10) | (metadata['Multi_Cell_Image_Id'] == 1305)]
multi_cell_images_names = metadata_subset['Multi_Cell_Image_Name'].unique()

print("Extracting metadata from multi cell images (subfolders):")
print(multi_cell_images_names)

metadata_small = filter_metadata_by_multi_cell_image_names(metadata, multi_cell_images_names)

create_directory("C:/git/generative-models-for-phenotypic-profiling/data/small/")
save_metadata(metadata_small, "C:/git/generative-models-for-phenotypic-profiling/data/small/metadata.csv")

print("completed!")
