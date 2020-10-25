model AdvectionScheme
  Modelica.Blocks.Sources.TimeTable data(
  table=[0,0; 1,0; 1,1; 4,1; 4,-1; 8,-1; 8,1; 12,1; 16,1; 16,0]);
  parameter Integer N = 100 "domain discretization";
  parameter Real c[3] = {0, 0, 0} "#optimize";
  parameter Real L = 1, v = 0.1, rel_v = v * N / L, dt = L / v;
  Real Y[N+1](each start = 0);
  Real y_true = delay(Y[1], dt) "#plot", y_sol = Y[end] "#plot";
  Real loss = abs(y_true - y_sol)^2 "#objective";
equation
  Y[1] = data.y;
  der(Y[2]) = rel_v * (Y[1] - Y[2]);
  der(Y[3:N]) = {rel_v * {c[1], -sum(c), c[2], c[3]} * Y[i - 2:i + 1] for i in 3:N};
  der(Y[N + 1]) = rel_v * (Y[N] - Y[N + 1]);
  annotation(experiment(StopTime = 30, Interval = 0.1));
end AdvectionScheme;
