# Load Forecasting for the average electrical consumption in the households in England

The goal of this project is to create a model to predict(24h ahead) the electrical consumption of one household. The model we use is a deep neural network.

The original dataset is preprocessed and consists of the electrical consumption of a household in England on a particular date and time (every 30 minutes).

The chosen **input parameter** for the neural network are: 
- **hour**
- **minute**
- **day of the week**
- **work day**
- **previus day at the same time electrical consumption**
- **previus week at the same time electrical consumption**
- **the average electrical consumption in the last 24h**

With these parameters the **MAPE** is **68.63%**.
