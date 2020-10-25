package HeatTransfer
  model TransientWall
    parameter Integer N = 20;
    parameter Real T_init = 20, L = 0.01, dx = L / N;
    parameter Real cp = 450, k = 50, rho = 7850, M=cp*rho*dx, K=k/dx;
    Real T[N + 1](each start = T_init), Qlbc "#plot", Qrbc = 20*(T[end] - T_init);
    Real T_sensor = T[end];
  equation
    M * der(T[1]) = Qlbc - K * (T[1]-T[2]);
    M * der(T[2:end-1]) = K * (T[1:end-2] - 2*T[2:end-1] + T[3:end]);
    M * der(T[end]) = K * (T[end-1]-T[end]) - Qrbc;
  end TransientWall;
  
  model InverseProblem
    Modelica.Blocks.Sources.TimeTable data(table = [0, 0; 20, 0; 20, 
    3000; 100, 3000; 100, -3000; 150, -3000; 150, 0; 300, 1300; 300, 
    2000; 400, 2000; 400, -4000; 500, -4000; 500, 0; 600, 1000]);
    TransientWall true_sim(Qlbc = data.y);
    constant Integer Nt = 40;
    constant Real t[Nt] = linspace(0, 800, Nt);
    parameter Real Qp[Nt](each min = -10000, each max = 10000) = fill(0, Nt) "#optimize";
    TransientWall sim(Qlbc = Modelica.Math.Vectors.interpolate(t, Qp, time));
    Real loss = abs(true_sim.T_sensor - sim.T_sensor) "#objective";
    annotation(experiment(StopTime = 800, Interval = 2));
  end InverseProblem;
end HeatTransfer;
