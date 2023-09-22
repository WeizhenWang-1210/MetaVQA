
# Metadrive New Asset Adder

This repo specifically is to easilly add new asset and change their location/size/other metainfo interactively.

You can download asset from online as long as it is in `.glb` or `.gltf` format.

You can also use `Objaverse`, a dataset containing massive amount of asset that match our requirmenets.

- Note that assets in `Objaverse` are poorly annotated and classified, so you might need to manually filter them once you have your initial assets.


## Table of Contents

- [Installation](#installation)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)

## Installation

1. Cloning the repository with correct branch
```bash
git clone -b chenda_dev https://github.com/WeizhenWang-1210/MetaVQA.git
```
2. Create a conda environment
```bash
conda create -n Metavqa python=3.8
conda activate Metavqa
```
3. Install our version of MetaDrive (This is important, we haven't merge some of the files in the main metadrive repo)
```bash
cd metadrive && pip install -e .
```
4. Install the rest requriements
```bash
cd .. && pip install -e .
```

## File Descriptions

- `MetaVQA/asset/download_assets.py`: Download assets from `objaverse` with certain tag. Save asset path json
- `MetaVQA/asset/pbjverse_filter_asset.py`: Go over the asset downloaded from objaverse, decide whether to use them later, and add annotations to each asset.
- `MetaVQA/asset/show_mesh.py`: Helper function to display an asset using trimesh.
- `MetaVQA/asset/objverse_change_asset.py`: Interactive Asset parameters updater that display a new CAR asset in the environment and let you adjust the parameters like size interactively.
- `MetaVQA/asset/objverse_change_asset_static.py`: Same as above, but for other static objects.
- `MetaVQA/asset/objversse_change_asset_script.py`: Script to apply the above script to batch of new assets.
- `MetaVQA/asset/layout_static_obj.py`: Manually placed adjusted assets in a static scene, to generate static demo.
- `MetaVQA/asset/pedestrian.py`: Test Pedestrian manager that auto-generate pedestrian on sidewalk.
- `MetaVQA/metadrive/envs/test_asset_metadrive_env.py`: Adjusted env that can place new `car` asset and change its parameters in real time. Used for "objverse_change_asset.py"
- `MetaVQA/metadrive/manager/test_asset_agent_manager.py`: Adjusted agent manager that place the `car` and update its parameter interactively.
- `MetaVQA/metadrive/envs/test_pede_metadrive_env.py`: Adjusted env that use new sidewalk manager to spawn static object on sidewalk
- `MetaVQA/metadrive/manager/sidewalk_manager.py`: Adjusted manager that adds items (currently pedestrian) on the sidewalk.

## Usage

Here is a normal pipeline for downloading, filtering, and adding new assets into Metadrive:
- Using `MetaVQA/asset/download_assets.py` to download assets with certain tag, generating `object-paths-tag.json`
- Using `MetaVQA/asset/pbjverse_filter_asset.py` to go over downloaded assets and make annotation, generating `matched_uids_tag.json`
- Using `MetaVQA/asset/objversse_change_asset_script.py` to adjust the size, orientation, etc, of the assets, generating `tag-uid.json`.
- Using `layout_static_obj.py` to place assets into a static scene, or using `pedestrian.py` to test generating assets into the map.