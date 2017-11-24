# Speaker Differentiation using TEDLIUM2 data

### Data

http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus

### Python Dependencies

- tensorflow
- tflearn
- numpy
- librosa

### Description

- Using TEDLIUM2 corpus to create model for differentiating between a specific speaker and rest of the speakers.

- Input is the speaker ID (an index) to prepare data with target having one hot encoding of size 2. This is to categorise either it is the speaker or it is not.

### How to run,

- extract the files using https://github.com/braindead/deepspeech.pytorch/blob/master/data/ted.py

- Use `makeTedData.py` to extract spectrograms and make train and test data for feeding into the conv net.

- Conv net is defined in `ted_conv_tflearn.py`
