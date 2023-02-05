<font color="red">For the information on the competition from 2022, please see [this page](https://github.com/TeamFightingICE/FightingICE/tree/master/DareFightingICE).</font>

# FightingICE #

このリポジトリは、格闘ゲームAIの性能を競う国際大会に使用されている2D格闘ゲームFightingICEの開発用リポジトリです。<br>


### About FightingICE ###
FightingICEはComputational Intelligence and Games (CIG)公認の格闘ゲームAIの性能を競う国際大会 Fighting Game AI Competition (FTGAIC)に使用されている2D格闘ゲームです。  

### Important contents ###
- src: ソースコードを格納したディレクトリ  
- data: ゲーム内で使用されるリソースを格納したディレクトリ  
	- ai: ゲーム内で用いるAIのjarファイルを格納したディレクトリ  
	- aiData: AIが参照するデータファイル等を格納したディレクトリ  
	- characters: キャラクタの画像及びアクションのパラメータファイルを格納したディレクトリ  
	- graphics: 背景や波動拳等のキャラクター以外の画像を格納したディレクトリ  
	- sounds: BGM及びSEを格納したディレクトリ  
- lib: 起動に必要なライブラリを格納したディレクトリ  
- doc: Javadocを格納したディレクトリ  
- python: pythonでFightingICEを起動する際に用いられるファイルやAI等を格納したディレクトリ  
- jar: FightingICEの及びAI作成用のライブラリのjarファイルを格納したディレクトリ  
- multi_start_up_for_Deep: FightingICEをpython経由で複数起動する方法を記述したディレクトリ
- for_nonDelay: 遅れフレームの無いFightigICEの起動方法を記述したディレクトリ．DeepLearningの使用時にオススメ

### How to use ###
FightingICE公式サイトを参照してください。  
<http://www.ice.ci.ritsumei.ac.jp/~ftgaic/index.htm>






# DareFightingICE #

You can open the file "FightingICE_nonDelay_play.bat" to try to play the game.And use 
"FightingICE_nonDelay_pybot.bat" to train your ai for nodelay.
(press key "z" to Confirm, and use "z","x","c" attack, "space" jump , and "↑","↓","←","→" to control the character to move.)

# About the Blind-T #
This is a ai use BlindAI's architecture , but the part of getting audio features which use deeplearning network with Transformer 
encoder.(BlindAI use 1D-CNN/FFT/Mel-spectrogram to build the encoder.)

<font color="red">For the information on the competition from 2022, please see [this page](https://github.com/ChaoxJz/FightingICE/tree/master/DareFightingICE/SampleAI/Blind-T/paper).</font>



### Reproduce the trained ai of Blind-T ###
#### 1.All of data in [this page](https://github.com/ChaoxJz/FightingICE/tree/master/DareFightingICE/SampleAI/Blind-T). ####

#### 2.The trained ai model in FightingICE/tree/master/DareFightingICE/SampleAI/Blind-T/last_trained_model, and also in FightingICE/tree/master/DareFightingICE/SampleAI/Blind-T/ppo_pytorch/ppo_pytorch/checkpoints/trans/rnn/rnn_1_frame_256_mctsai/0 . ####
