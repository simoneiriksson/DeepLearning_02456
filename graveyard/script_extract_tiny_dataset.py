from utils.data_preparation import *

metadata = read_metadata("../data/all/metadata.csv")
metadata = drop_redundant_metadata_columns(metadata)

multi_cell_images_names = ["B02_s1_w1B1A7ADEA-8896-4C7D-8C63-663265374B72"]
metadata_small = filter_metadata_by_multi_cell_image_names(metadata, multi_cell_images_names)

create_directory("../data/tiny/")
save_metadata(metadata_small, "./data/tiny/metadata.csv")

print("completed!")