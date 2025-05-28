## License

This project is licensed under the GNU Affero General Public License, version 3 (AGPL-3.0-only).  
See the [LICENSE](LICENSE) file for details.  
![AGPL-3.0-only](https://img.shields.io/badge/License-AGPLv3-blue.svg)


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




## Dependencies & Acknowledgments

- **Flan-T5-Base** (Hugging Face Transformers)  
  [![Apache 2.0][apache-badge]](https://www.apache.org/licenses/LICENSE-2.0)  
  SPDX: `Apache-2.0`

- **SpatialBot** (BAAI-DCAI)  
  [![MIT][mit-badge]](https://opensource.org/licenses/MIT)  
  SPDX: `MIT`

- **Open3DSG** (Bosch Research)  
  AGPL-3.0-only  
  - Full text: https://www.gnu.org/licenses/agpl-3.0.en.html  
  SPDX: `AGPL-3.0-only`

- **moonshot-v1-8k API**  
  see [Moonshot Terms](https://api.moonshot.ai/terms) for usage and attribution requirements

### Data

- **3RScan**  
  [![MIT][mit-badge]](https://opensource.org/licenses/MIT)  
  SPDX: `MIT`

- **3DSSG**  
  [![CC BY-NC-SA 4.0][ccbyncsa-badge]](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
  SPDX: `CC-BY-NC-SA-4.0`

[apache-badge]:      https://img.shields.io/badge/License-Apache_2.0-blue.svg
[mit-badge]:         https://img.shields.io/badge/License-MIT-green.svg
[ccbyncsa-badge]:    https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg

