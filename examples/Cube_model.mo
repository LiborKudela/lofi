package Cube_model
  model Data
    parameter Real dummy_p = 1 "#optimize";
    parameter Real u0[2] = {2, 0};
    parameter Real A[2, 2] = {{-0.1, 2.0}, {-2.0, -0.1}};
    Real u[2] "#output";
  initial equation
    u = u0;
  equation
    der(u) = u .^ 3 * A;
  end Data;

  model NeuralODE
    parameter Real u0[2] = {2, 0};
    Real u[2] "#output";
    //NODE weights
    parameter Real W1[50, 2](each min = -5.0, each max = 5.0, each start = 0.0) "#optimize, #weight";
    parameter Real b1[50](each min = -5.0, each max = 5.0, each start = 0.0) "#optimize, #bias";
    parameter Real W2[2, 50](each min = -5.0, each max = 5.0, each start = 0.00) "#optimize, #weight";
    parameter Real b2[2](each min = -5.0, each max = 5.0, each start = 0.0) "#optimize, #bias";
  initial equation
    u = u0;
  equation
    der(u) = W2 * tanh(W1 * u .^ 3 + b1) + b2;
    
  end NeuralODE;

  model Training
    parameter Real u0[2] = {2,0};
    Cube_model.Data data(u0=u0);
    Cube_model.NeuralODE neuralODE(u0=u0);
    Real loss "#objective";
  equation
    loss = sum(abs(data.u - neuralODE.u));
    annotation(
      experiment(StartTime = 0, StopTime = 25, Tolerance = 1e-6, Interval = 0.05));
  end Training;
end Cube_model;