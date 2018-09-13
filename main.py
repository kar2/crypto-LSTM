import data
import model
import matplotlib.pyplot as plt

def plot_predictions():
    train = model.train(data.num_train_steps)
    test = model.test(data.num_test_steps)
    plt.plot(model.y_test, label="True Value")
    plt.plot(test['Predictions'], label="Prediction")
    plt.title("Bitcoin Price Change Prediction for %s Time Steps"  % (data.num_test_steps))
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Price Change")
    plt.legend()
    plt.savefig('results.jpg')

if __name__ == '__main__':
    plot_predictions()
