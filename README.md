


# Embodied Scene Understanding for Vision Language Models via MetaVQA 
<!---
[![build](https://github.com/metadriverse/metadrive/workflows/test/badge.svg)](http://github.com/metadriverse/metadrive/actions)
[![Documentation](https://readthedocs.org/projects/metadrive-simulator/badge/?version=latest)](https://metadrive-simulator.readthedocs.io)
[![Downloads](https://static.pepy.tech/badge/MetaDrive-simulator)](https://pepy.tech/project/MetaDrive-simulator)
-->
[![GitHub license](https://img.shields.io/github/license/metadriverse/metadrive)](https://github.com/metadriverse/metadrive/blob/main/LICENSE.txt)
[![GitHub contributors](https://img.shields.io/github/contributors/metadriverse/metadrive)](https://github.com/WeizhenWang-1210/MetaVQA/graphs/contributors)



<div style="text-align: center; width:100%; margin: 0 auto; display: inline-block">
<strong>
[
<a href="https://metadriverse.github.io/metavqa/">Website</a>
|
<a href="https://metadriverse.github.io/metaVQA/">Data Preview(coming)</a>
|
<a href="https://arxiv.org/abs/2501.09167">Arxiv</a>
|
<a href="https://metadriverse.github.io/">Relevant Projects</a>
]
</strong>
</div>


<!---
![](documentation/source/figs/metavqa_teaser.png)
-->
<br>

[Weizhen Wang](https://github.com/WeizhenWang-1210), [Chenda Duan](https://chendaduan.com/), [Zhenghao Peng](https://pengzhenghao.github.io/), [Yuxin Liu](https://github.com/Yuxin45), [Bolei Zhou](https://boleizhou.github.io/)



# Installation Guide
- Clone the repository and create a virtual environment/Conda envrionment with Python 3.11
- install the dependencies by running `pip install -e .`
- install the `adj_parameter` zip file, and unzip it at specified location `<path to parameter>`
- Spefified the path to the `adj_parameter` zip file in `path_config.yaml`. Overwrite `metadriveasset` with the absolute path at which the assets are downloaded and `parentfolder` with `<path to parameter>`

# Model Checkpoints & Benchmark Reproduction

# Model Training and Inference

# Citation



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




## Repository Update Timeline
- [x] Release of Demo VQA dataset(downloadable in the <a href="https://metadriverse.github.io/metaVQA/">official Website</a>)
- [ ] Release of MetaVQA-2M dataset
- [x] Demo for generating new VQA data
- [ ] Release of benchmark models and checkpoints.
- [x] Release of benchmark training repository.
- [ ] Setup of leaderboard website.

## MetaVQA-2M Dataset

We will release MetaVQA-2M in the <a href="https://metadriverse.github.io/metaVQA/">official Website</a>



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



## Acknowledgements
MetaVQA is built on top of <a href="https://github.com/metadriverse/metadrive">MetaDrive</a> simulator. Safety-critical scenarios
are generated using <a href="https://github.com/metadriverse/cat">CAT</a>. 


-->





