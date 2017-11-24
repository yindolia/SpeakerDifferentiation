# Speaker Differentiation using TEDLIUM2 data

### Data

http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus

### Python Dependencies

- tensorflow
- tflearn
- numpy
- librosa

# Description

- Using TEDLIUM2 corpus to create model for differentiating between a specific speaker and rest of the speakers.

- Input is the speaker ID (an index) to prepare data with target having one hot encoding of size 2. This is to categorise either it is the speaker or it is not.

