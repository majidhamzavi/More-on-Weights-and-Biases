# More-on-Weights-and-Biases
In this topic, I want to generate random sets of weights and biases for the first layers and the last layer's weight and bias would be obtained directly from <img src="https://render.githubusercontent.com/render/math?math=Y_{train}">.

Here, we randomly generate w0, b0, w1 and b1. For w2 and b2 we use y. As you remember, the last layer is computed as <img src="https://render.githubusercontent.com/render/math?math=Out = \sigma(H1 * w2 %2B b2)">, where <img src="https://render.githubusercontent.com/render/math?math=\sigma"> is sigmoid rectifier. For the last year, w2 can be calculated as: <img src="https://render.githubusercontent.com/render/math?math=w2 = pseudoinverse(H1)(\textit{logit}(Y_{train}) - b2)"> where <img src="https://render.githubusercontent.com/render/math?math=b2 = mean(Y_{train} * 0.9 %2B 0.05)"> to keep the arguments of the logit function in a valid range. 

Accuracy: 91.3%, how awsome is that with very simple math!

