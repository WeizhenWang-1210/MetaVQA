# Embodied Scene Understanding for Vision Language Models via MetaVQA 
[![GitHub license](https://img.shields.io/github/license/metadriverse/metadrive)](https://github.com/metadriverse/metadrive/blob/main/LICENSE.txt)
[![GitHub contributors](https://img.shields.io/github/contributors/metadriverse/metadrive)](https://github.com/WeizhenWang-1210/MetaVQA/graphs/contributors)



<div style="text-align: center; width:100%; margin: 0 auto; display: inline-block">
<strong>
[
<a href="https://metadriverse.github.io/metavqa/">Website</a>
|
<a href="https://huggingface.co/datasets/Weizhen011210/MetaVQA-Train">Data Preview</a>
|
<a href="https://arxiv.org/abs/2501.09167">Arxiv</a>
|
<a href="https://metadriverse.github.io/">Relevant Projects</a>
]
</strong>
</div>

<br>

[Weizhen Wang](https://github.com/WeizhenWang-1210), [Chenda Duan](https://chendaduan.com/), [Zhenghao Peng](https://pengzhenghao.github.io/), [Yuxin Liu](https://github.com/Yuxin45), [Bolei Zhou](https://boleizhou.github.io/)

This is the official repository for **Embodied Scene Understanding for Vision Language Models via MetaVQA** from CVPR 2025. It contains the necessary toolkit for creating this benchmark, including both VQA datasets and closed-loop challenges. 

# Installation
Clone the repository and create a virtual environment/Conda envrionment with Python 3.11
```bash
$ git clone 
$ cd MetaVQA
$ conda create -n metavqa python=3.11 -y
$ conda activate metavqa
```
install the metadrive dependencies by running 
```bash
$ pip install -e .
```

MetaVQA needs some extra pacakges. You can use
```bash
$ pip install PyYAML
$ pip install imageio[ffmpeg]
```
Once the previous steps are finished, use the following for installation verification
```bash
$ python -m metadrive.examples.drive_in_single_agent_env
```

For visually diverse simulation envrionments, download and unzip the `asset_v0.0.4.zip` and `adj_parameter_folder_v0.0.4.zip` from [this link](https://github.com/WeizhenWang-1210/MetaVQA/releases). Move the `test` folder within `asset_v0.0.4.zip` into `metadrive/assets/models`. You need to pull the vanilla metadrive asset first, and **this will be automatically done when you do verification**. 

You should have the following file structure
```
-MetaVQA
    -metadrive
        -assets
            -models
                -test/*
```

Lastly, modify the `path_config.yaml` by overwriting 
```yaml
...
 # Specify location of the asset within metadrive, download "asset-model.zip" from github release and put it at corresponding location.
metadriveasset: <absolute path to MetaVQA's parent folder>/MetaVQA/metadrive/assets/models/test
# The parent path for the subfolders below
parentfolder: <absolute path to the parameter folder>/adj_parameter_folder_v0.0.4
...
```
You can verify the installation of additional assets by running
```bash
$ python -m metadrive.examples.drive_in_real_env_diverse
```

## Preparation of the nuScenes Dataset
As part of the MetaVQA-Dataset leverage nuScenes Dataset, we provide a brief tutorial to set it up.
Go to the [nuScenes official webpage](https://www.nuscenes.org/nuscenes) download the dataset. Additional, this website provides details on the dataset composition.

Much of the data collection is done using the `nuScenes-Devkit`. We recommend starting a dedicated virtualized environments:
```bash
$ conda create -n nusc python=3.7 -y
$ conda activate nusc
$ pip install nuscenes-devkit
```
In case of confusion, check out the devkit's implementation [here](https://github.com/nutonomy/nuscenes-devkit)

# Scenario Aggregation
We prepared two distinct pipelines for curating the real-world scenarios and simulator-rendered scenarios. Checkout `scripts_cvpr/scene_gen/nusc_real.sh` and `scripts_cvpr/scene_gen/waymo_sim.sh` for examples.

## Preliminaries
Please download the [nuScenes Dataset](https://www.nuscenes.org/nuscenes) via the official website if you want to utilize nuScenes scenarios for VQA generation. For Waymo Open Motion Dataset(WOMD), you can refer to [ScenarioNet](https://github.com/metadriverse/scenarionet) to pre-process the tfrecords into `pkl` files compatible with [MetaDrive](https://github.com/metadriverse/metadrive) simulator.

## nuScenes Scenarios (with real-image Observations)
We utilize the nuScenes_Devkit tools to prepare nuScenes scenarios. You can checkout `vqa/scenegen/nusc_devkit_annotation.py` for implementation details. Suppose you download the nuScenes dataset at `nusc` folder, you should overwrite `vqa/scenegen/macros.py` with
```python
NUSC_PATH = 'nusc'
NUSC_VERSION = 'v1.0-trainval'
```
You will have output in the following structure
```
nusc_scenarios/
    |-scene-0510_0_40/
    |               |
    |               |-1_0/
    |               |    |-world_1_0.json (the recorded scene graph)
    |               |    |-rgb_front_1_0.json (CAM_FRONT RGB image)
    |               |    |-mask_front_1_0.png (Instance Segmentation Mask, in boxes)
    |               |    |id2corner_1_0.json (Map an object id in the scene graph to a 2D pixel coordinates)
    |               |    |id2c_1_0.json (Map an object id to an instance color, rounded to 5 in float)
    |               |-1_1
    |               |...
    |               |-<seed_id>_<keyframe_idx>
    |-scene-0515_0_40/*
    |...      
    |-<scene_id>_<keyframe_start>_<keyframe_end>

```

## WOMD Scenarios (with simulator-rendered Observations)
Check out `vqa/scenegen/metadrive_annotation.py` for more details. 


Note that nuScenes scenarios can also be converted to this format, and you can essentially create a "digital twin" for the same traffic layout. Check out `vqa/scenegen/nusc_metadrive_annotation.py` for paired aggregation. Note that you have to first collect camera data in order to use it, see `vqa/scenegen/macros.py`:
```python
PAIRING_PATH = "/bigdata/weizhen/data_nusc_multiview_trainval.json"
NUSCENES_SN_PATH = "/bigdata/datasets/scenarionet/nuscenes/trainval"
```

<!--
## End Product #TODO
By the end of this stage, suppose you specify your logging path at `/data`, you will have the following file structure.
-->



# VQA Curation
## Set-of-Mark Annotation:
Checkout `vqa.vqagen.set_of_marks.py` for implementation details.

## VQA Generation
Checkout `scripts_cvpr/vqa_gen/*.sh` for sample code and `vqa/vqa_gen/static_question_generation.py` for implementation details. All the question templates are defined in `vqa/vqa_gen/questions_templates.json`. Note that the Set-of-Mark annotation is automatically applied during the generation process.



# MetaVQA-Datasets
The Training-Validation and Testing sets used in our CVPR 2025 paper have been released on Hugginface in JSON files. Check the table below. 

| Split     | URL       |  Size (#VQAs)
|---------  |:---------:|:---------:|
| Train-Val | https://huggingface.co/datasets/Weizhen011210/MetaVQA-Train  | 150 K
| Test      | https://huggingface.co/datasets/Weizhen011210/MetaVQA-Eval   | 9,375


A much larger version will be released soon.


# Closed-loop Evaluation (Coming soon)



# Fine-tuned Checkpoints 
You can find fine-tuned checkpoints for models referred in the paper here. You can simply load them using the [Transformers](https://huggingface.co/docs/transformers/en/index) package and inference in consistent paradigm their base models.

| Table       | Code Name       |  URL
|:---------:  |:---------:|:---------:|
| 4           | Qwen2-finetuned      | https://huggingface.co/Weizhen011210/Qwen2-VL_MetaVQA-Train 
| 4           | Llama3.2-finetuned   | https://huggingface.co/Weizhen011210/Llama3.2_MetaVQA-Train 
| 5           | Qwen2-tuned          | https://huggingface.co/Weizhen011210/Qwen2-VL_MetaVQA-Closed-Loop
| 5           | Llama3.2-tuned       | https://huggingface.co/Weizhen011210/Llama3.2_MetaVQA-Closed-Loop
| 5           | InternVL2-4B-tuned   | https://huggingface.co/Weizhen011210/InternVL2-4B_MetaVQA-Closed-Loop
| 5           | InternVL2-8B-tuned   | https://huggingface.co/Weizhen011210/InternVL2-8B_MetaVQA-Closed-Loop





<!---

# Benchmark Reproduction
## VQA Evaluation Results
For transparency, you can find the original VLM inference results and our computed metric files listed below.

### Table 2. Sim-to-Real



## Closed-loop Evaluation Results
For transparency, we provide the closed-loop inference results below.




# Model Fine-tuning and Inference

## Supervised Fine-tuning
We used InternVL2's native repository for the LoRA fine-tuning of InternVL2. For Qwen-2-VL and Llama-3.2-VL, we used LLaMa-Factory. For inference, we used the [Transformers](https://huggingface.co/docs/transformers/en/index) package (except for GPT-4o, which we uses web API).

## VQA Inference

## Closed-loop Inference
-->

# Acknowledgements
MetaVQA is built on top of <a href="https://github.com/metadriverse/metadrive">MetaDrive</a> simulator. Safety-critical scenarios
are generated using <a href="https://github.com/metadriverse/cat">CAT</a>. 



# Citation
If you find our work useful, please cite as follows:

```latex
@inproceedings{wang2025metavqa,
  title={Embodied Scene Understanding for Vision Language Models via MetaVQA},
  author={Wang, Weizhen and Duan, Chenda and Peng, Zhenghao and Liu, Yuxin and Zhou, Bolei},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}
```

<!---
## Highlights <a name="highlights"></a>
MetaVQA a visual question-answering benchmark for improving and evaluating the embodied scene understanding of VLMs.
    
* MetaVQA designs a scalable pipeline to generate visual question answer (VQA) pairs relating to traffic scenarios imported from various sources, including nuScenes dataset, Waymo Open Motion Dataset, and a synthetic dataset of safety-critical scenes.

* MetaVQA provides a large-scale VQA dataset(MetaVQA-2M) containing 2.7M questions for 291K frames related to spatial, visual, dynamic, and safety-critical counterfactual scene understandings.

* MetaVQA establishes the baseline performance of VLMs on the dataset and show that the VLMs achieve remarkable embodied scene understanding capabilities through instruction tuning, especially when handling safety-critical situations.

## News <a name="news"></a>
- `[2024/07/02]` We provide an example for the holistic VQA generation pipeline in this [section](#vqa-generation).
- `[2024/07/01]` Training scripts for the benchmarks made public <a href="https://github.com/Dadaism6/MetaVQA-Training">here</a>. Demo dataset is released in the <a href="https://metadriverse.github.io/metaVQA/">official Website</a>(517 GB)
- `[2024/06/27]` MetaVQA repository made public.








## Dataset Structure

Dataset exported from MetaVQA will have the following file structure.
```
-root
    -episodes
        -episode0
            -frame0
                -front.png
                -left.png
                ...
            -frame1
            ...
        -episode1
            ...
        ...
    -data.json
    -mapping.json
```
To load the dataset, simply load `data.json`. This json file have the following structure
```
-data
    -1
        -question: ...
        -answer: ...
        -rgb: ...
    -2
        -question: ...
        -answer: ...
    ...
-split
    -train: [1,2,3,...]
    -val: ...
    -test: ...
```
The `rgb` field contains a list of path to the observations(ordered chronologically) for each view angle.


## VQA Generation
By default, MetaVQA generates VQA data on real-world traffic in <a href="https://metadriverse.github.io/scenarionet/">ScenarioNet</a> format. Suppose
you have the traffic data stored in `<traffic_folder>`, and you want the collected episodes to be stored in `<episodes_folder>` with `<num_proc>` processes.
First, run
```
python -m vqa.multiprocess_episodes_generation \
        --num_proc <num_proc>                  \
        --scenarios <traffic_folder>           \
        --data_directory <episodes_folder>
```
Optionally, you can set the `--headless` flag to boost performance. 

To generate questions and store them in `<questions_folder>`, you first need to compose a config file(`<vqa_config.yaml>`). Here is an example:
```
num_proc: 32                               # number of process to use
root_directory: <episodes_folder> 
output_directory: <questions_folder> 
src: "NuScenes"                              # traffic source
verbose: False                             # Set true for verbose output.
```
Then, run
```
python -m vqa.multiprocess_question_generation --job <job> --config <vqa_config.yaml>
```
You can choose `<job>` from `[static, dynamic, safety, static_nusc, dynamic_nusc, safety_nusc]`. 
Upon completion, `<question_folder>` will have `<num_proc>` jsons files with name in form `<job>_<proc_id>.json`

Then, preprocess the json files containing vqa data from the previous steps and store the processed data into `<processed_folder>` by 
```
python -m vqa.qa_preprocessing --raw <questions_folder> --processed <processed_folder>
```

Finally, to export the dataset into a self-contained folder, with `export.py`. The exported dataset will have the same layout as in the [previous section](#dataset-structure).




## Benchmark Training and Testing
The training and evaluation scripts are available at  <a href="https://github.com/Dadaism6/MetaVQA-Training">this repository</a>. You can find the experiment
results in the <a href="https://metadriverse.github.io/metaVQA/">official Website</a>.


-->





