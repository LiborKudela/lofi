model LotkaVolteraODE
  // trainnig data
  parameter Real a = 1.5, b = 1.0, c = 3.0, d = 1.0;
  Real x_data(start=1) "#plot", y_data(start=1) "#plot";
  // model stuff
  parameter Real p[4](min=-5.0, max=5.0) "#optimize";
  Real x_model(start=1) "#plot", y_model(start=1) "#plot";
  Real loss = sum((u_true-u).^2) "#objective";
  equation
    //training data
    der(x_data) = a*x_data - b*x_data*y_data;
    der(y_data) = -c*y_data + d*x_data*y_data;

    //model
    der(u[1]) = p[1]*u[1] - p[2]*u[1]*u[2];
    der(u[2]) = -p[3]*u[2] + p[4]*u[1]*u[2];

annotation(experiment(StopTime = 10, Interval = 0.01));
end LotkaVolteraODE;
