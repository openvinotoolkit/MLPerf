# MLPerf Intel OpenVino Inference Code

### Environment
*  Ubuntu 20.04
*  cmake 3.23.2
*  gcc/g++ 9.4.0
*  Python 3.8.10
*  [MLPerf™ Inference Benchmark Suite v3.0](https://github.com/mlcommons/inference/tree/v3.0)
*  [OpenVINO Toolkit 2023.0](https://github.com/openvinotoolkit/openvino/tree/releases/2023/0)

### Rules
MLPerf Inference Rules are [here](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc).

### Supported benchmarks

| Area     | Task                       | Model                                                                      | Dataset                      | SingleStream       | MultiStream        | Server             | Offline            |
|----------|----------------------------|----------------------------------------------------------------------------|------------------------------|--------------------|--------------------|--------------------|--------------------|
| Vision   | Image classification       | [Resnet50-v1.5](https://zenodo.org/record/4735647/) (Image classification) | ImageNet (224x224)           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Vision   | Object detection           | [Retinanet](https://zenodo.org/record/6605272) (Object detection)          | OpenImages (800x800)         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Vision   | Medical image segmentation | [3D UNET](https://zenodo.org/record/5597155) (Medical image segmentation)  | KITS 2019 (602x512x512)      | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: |
| Language | Language processing        | [BERT-large](https://zenodo.org/record/3733910) (Language processing)      | SQuAD v1.1 (max_seq_len=384) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

:heavy_check_mark:  - supported
:x: - not supported

### Execution Modes
*  Performance
*  Accuracy
    + User first runs the benchmark in ```Accuracy``` mode to generate ```mlperf_log_accuracy.json```
    + User then runs a dedicated accuracy tool provided by MLPerf

### BKC on CPX, ICX systems
Use the following to optimize performance on CPX/ICX systems.  These BKCs are provided in `performance.sh` mentioned in [How to Build and Run](#how-to-build-and-run).
 - Turbo ON
   ```
   echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
   ```
 - Set CPU governor to performance (Please rerun this command after reboot):
    ```
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    ```
    OR
    ```
    cpupower frequency-set -g performance
    ```

### How to Build and Run
1. Navigate to root repository directory. This directory is your **BUILD_DIRECTORY**.
2. Run the build script:
   ```
   ./build.sh
   ```
   ***NOTE: sudo privileges are required***
3. Modify ***BUILD_DIRECTORY*** in ```setup_env.sh```(if necessary) and source:
    ```
    source scripts/setup_env.sh
    ```
4. Run the performance script for CPX/ICX systems:
   ```
   ./scripts/performance.sh
   ```
5. Download models
    ```
    ./scripts/download_models.sh [specific model]
    ```
6. Download datasets
   ```
   ./scripts/download_datasets.sh [specific dataset]
   ```
7. Modify script ```./scripts/run.sh``` to apply desired parameters.

   The following OpenVINO parameters should be adjusted based on selected hardware target:
   * number of streams
   * number of infer requests
   * number of threads
   * inference precision
8. Update MLPerf parameters (`user.conf` and `mlperf.conf`) if it is needed.
9. Run:
    ```
    ./scripts/run.sh -m <model> -d <device> -s <scenario> -e <mode>
    ```
    For example (results will be stored into the `${BUILD_DIRECTORY}/results/resnet50/CPU/Performance/SingleStream` folder):
    ```
    ./scripts/run.sh -m resnet50 -d CPU -s SingleStream -e Performance
    ```
   To run all combination of models/devices/scenarios/modes:
    ```
    ./scripts/run_all.sh
    ```
    ***NOTE: This product is not for production use and scripts are provided as example. For reporting MLPerf results dedicated scripts should be provided for each model with suitable parameters.***
