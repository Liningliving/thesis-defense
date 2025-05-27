## The workflow of the project is as follows:
# Step 1: Select frames
In this step, we use code to select frames from the whole scene, in consideration of there are many vague frames and almost duplicated frames.

# Step 2: Get object frames
These codes are from open3DScan repository.

# Step 3: Describe the object frames and summarize them
Describe objects from the object frames using SpatialBot and summarize descriptions using FLAN-T5 model.

## To run this you need to
Download the 3RScan dataset (actually only .ply and .zip are be used) and 3DSSG_subset.zip.

# run to Get object frames


## About choose from rgb align with depth or depth align with rgb?
This is depend on the resolution of these two kinds of data.
If the difference is too large, choose to projected the higher resolution data to align with the low resolution data.
For instance, in this data set 

## As for using faiss-gpu, I need some extra package, but I don't have sudo pwd.