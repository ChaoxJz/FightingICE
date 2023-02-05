# Deep Reinforcement Learning Blind-T

This is a ai use BlindAI's architecture , but the part of getting audio features which use deeplearning network with Transformer 
encoder.(BlindAI use 1D-CNN/FFT/Mel-spectrogram to build the encoder.)

<font color="red">For the information on the competition from 2022, please see [this page](https://github.com/ChaoxJz/FightingICE/tree/master/DareFightingICE/SampleAI/Blind-T/paper).</font>

## Installation:
- Install miniconda for python 3.8: https://docs.conda.io/en/latest/miniconda.html.
- Clone the repo: `git clone https://github.com/ChaoxJz/FightingICE
- Create and activate conda env:
  
    ```
    cd DareFightingICE/DareFightingICE/SampleAI/Blind-T
    conda env create -f environment.yml
    conda activate ice
    ```

- Merge all the java files under ```non_delay``` with the source code of DareFightingICE under ```src/aiinterface```.
- Boot DareFightingICE with the option "--py4j --limithp 400 400".
- Run the ```train.py``` file to train. e.g ```python torch_train.py --p2 MctsAi65 --encoder trans --id rnn_1_frame_256_mctsai65 --n_frame 1 --recurrent``` or
- Run the ```trained_ai/PvJ.py``` to test the Blind AI.

## Reproduce the trained ai of Blind-T ##
#### 1.All of data in this folder. ####

#### 2.The trained ai model in FightingICE/tree/master/DareFightingICE/SampleAI/Blind-T/last_trained_model, and also in FightingICE/tree/master/DareFightingICE/SampleAI/Blind-T/ppo_pytorch/ppo_pytorch/checkpoints/trans/rnn/rnn_1_frame_256_mctsai/0 . ####


## File Description
- ```train.py``` is a file used to train Blind AI. Please run ```python train.py -h``` for our explanation of the parameters.
- ```train_ai``` folder contains source code for the trained AI which is used as a sample AI for the AI track. It uses the FFT audio encoder and GRU. All weights are stored in ```trained_model```.
- ```non_delay``` folder contains modifications of ```AIController.java``` and ```AIInterface.java```. They make DareFightingICE with no frame delay.
- ```visualize.py``` is used to visualize the learning curve and calculate the area under the learning curve.
- ```analyze_fight_result.py``` is used to calculate the win ratio and average HP difference between Blind AI and MctsAi65.

## Get sound design evaluation metrics
- After finishing your sound design, please run the following command to train Blind AI:
  ```python torch_train.py --p2 MctsAi65 --encoder fft --id {experiment_id} --n_frame 1 --recurrent```, where you can decide ```experiment_id``` on your own
- After training, a result file with the name ```result_fft_{experiment_id}_rnn.txt``` is created. Please run ```visualize.py``` as follows: ```python visualize.py --file result_fft_{experiment_id}_rnn.txt --title FFT```. A plot will be shown and the area under the learning curve will be printed out.
- Before testing the performance of the Blind AI against MctsAI65, please remove all the files under ```log/point``` of DareFightingICE.
- Please revise the ```path``` parameter of the Blind AI in line number 22 of ```trained_ai/PvJ.py``` to your trained model location.
- Run ```trained_ai/PvJ.py``` to begin testing.
- After testing, please run ```analyze_fight_result.py```. Both win ratio and average HP difference will be printed out.
## Tested Environment
- OS：Windows 10 professional
- ハードディスク
	- GPU：NVIDIA GeForce GTX 1080 8GBytes
	- CPU：Intel Core i7-6950X @ 3.00GHz
	- メモリ：128 GBytes DDR4
	- マザーボード：ASUS X99-E WS
- ゲーム：DareFightingICE 5.2 (from the project's Github)
- CUDA 12.0.89
- CUDA Toolkit
- Condaの環境 
	- Python 3.8.13
	- ライブラリ
		- altair
		- spacy
		- GPUtil
		- warnings
		- pytorch 1.11.0
		- torchaudio 0.11.0
		- torchvision 0.12.0

