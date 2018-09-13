[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
# Cryptocurrency Price Prediction with Recurrent Neural Network/LSTM

This is a recurrent neural network utilizing LSTM to predict time-series Bitcoin pricing data. Built using Tensorflow's mathematical operators to represent a singular LSTM cell then manually entering input data, previous output data, and previous cell state data for each time step.

## How to Use

Run `main.py` to train and test data. The number of training steps and number of testing steps to predict can be customized in `data.py`.

```
python3 main.py
```
## Results

The following prediction was trained on 25000 time steps and tested on 250 time steps.
![Results](https://user-images.githubusercontent.com/9154924/45461848-0478d480-b6ba-11e8-8f8f-320f257e664d.jpg)

## Contributing

Feel free to fork this repository and make changes.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Reference
[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
