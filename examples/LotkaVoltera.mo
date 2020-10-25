package LotkaVoltera
  model LV_data "Simulates known system to get data"
    constant Real a = 1.5, b = 1.0, c = 3.0, d = 1.0;
    Real x(start = 1) "#plot", y(start = 1) "#plot";
  equation
    der(x) = a * x - b * x * y;
    der(y) = -c * y + d * x * y;
  end LV_data;

  model ODE
    LV_data data;
    parameter Real p[4](each min = 0, each max  = 5, each start = 0.5) "#optimize";
    Real x(start = 1) "#plot", y(start = 1) "#plot";
    Real loss = (data.x - x)^2 + (data.y - y)^2 "#objective";
  equation
    der(x) = p[1] * x - p[2] * x * y;
    der(y) = -p[3] * y + p[4] * x * y;
    annotation(experiment(StopTime = 10, Interval = 0.05));
  end ODE;

  model NeuralODE
    LV_data data;
    parameter Integer N = 50 "number of neurons in hidden layer";
    parameter Real W1[N,2](each start = 0.1) "weights #optimize";
    parameter Real b1[N](each start = 0.1) "biases #optimize";
    parameter Real W2[2,N](each start = 0.1) "weights #optimize";
    parameter Real b2[2](each start = 0.1) "biases #optimize";
    Real x(start = 1) "#plot", y(start = 1) "#plot";
    Real loss = abs(x - data.x) + abs(y - data.y) "#objective";
  equation
    der({x,y}) = W2*tanh(W1*{x,y}+b1)+b2;
    annotation(experiment(StopTime = 10, Interval = 0.05));
  end NeuralODE;
end LotkaVoltera;
