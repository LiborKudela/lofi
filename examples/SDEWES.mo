package SDEWES
  package Advection
    model SeccondOrderScheme
      //input output interface
      Modelica.Blocks.Interfaces.RealInput y_input "Input signal" annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput y_output = Y[end] "Output signal" annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      // advection scheme hyperparameters
      parameter Integer N = 20 "Number of elements";
      parameter Real c[3] = {0, -1, 0} "2nd order scheme coefficients #optimize";
      parameter Real op[4] = {c[1], -sum(c), c[2], c[3]} "actual diff operator";
      // geometry stuff
      parameter Real L;
      parameter Real v;
      parameter Real relative_velocity = v / (L / N);
      Real Y[N];
    initial equation
      Y = fill(0.0, N);
    equation
      der(Y[1]) = relative_velocity * (y_input - Y[1]);
      der(Y[2]) = relative_velocity * op*{y_input,Y[1],Y[2],Y[3]};
      for i in 3:N - 1 loop
        der(Y[i]) = relative_velocity * op*Y[i-2:i+1];
      end for;
      der(Y[N]) = relative_velocity * (Y[N - 1] - Y[N]);
      annotation(
        defaultComponentName = "solver",
        Icon(graphics = {Rectangle(fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid, lineThickness = 0.75, extent = {{-100, 100}, {100, -98}}), Text(origin = {-2, 41}, extent = {{-92, 51}, {100, -41}}, textString = "AdvectionSolver")}, coordinateSystem(initialScale = 0.1)));
    end SeccondOrderScheme;

    model NeuralScheme
      // input output interface
      Modelica.Blocks.Interfaces.RealInput y_input "Input signal" annotation(
        Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-100, 0}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
      Modelica.Blocks.Interfaces.RealOutput y_output = Y[end] "Output signal" annotation(
        Placement(visible = true, transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
      // advection scheme hyperparameters
      parameter Integer N = 20 "Number of elements";
      //Neural net size, weights and biases
      parameter Real W1[10, 4](each min = -1, each max = 1, each start = 0.1) "#optimize";
      parameter Real b1[10](each min = -3, each max = 3, each start = 0.1) "#optimize";
      parameter Real W2[1, 10](each min = -1, each max = 1, each start = 0.1) "#optimize";
      //parameter Real b2[1](each min = -3, each max = 3, each start = 0.1) "#optimize";
      // geometry stuff
      parameter Real L;
      parameter Real v;
      parameter Real relative_velocity = v / (L / N);
      Real Y[N];
    initial equation
      Y = fill(0.0, N);
    equation
      der(Y[1]) = relative_velocity * (y_input - Y[1]);
      der(Y[2:2]) = relative_velocity * W2*tanh(W1*{y_input,Y[1],Y[2],Y[3]}+b1);
      for i in 3:N - 1 loop
        der(Y[i:i]) = relative_velocity * W2*tanh(W1*Y[i-2:i+1]+b1);
      end for;
      der(Y[N]) = relative_velocity * (Y[N - 1] - Y[N]);
     annotation(
        defaultComponentName = "solver",
        Icon(graphics = {Rectangle(fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid, lineThickness = 0.75, extent = {{-100, 100}, {100, -98}}), Text(origin = {-2, 41}, extent = {{-92, 51}, {100, -41}}, textString = "AdvectionSolver")}, coordinateSystem(initialScale = 0.1)));
    end NeuralScheme;
    
    model GeneralTestBench
      parameter Real L = 1 "Length of the domain (using unit length for conveniece)";
      parameter Real v = 0.1 "Advection velocity";
      parameter Modelica.SIunits.Time delay_time = L / v;
      Real y_exact = delay(solver.y_input, delay_time) "Exact solution #plot";
      Real y_aprox = solver.y_output "Optimised solution #plot";
      Real loss = abs(y_exact - solver.y_output) "#objective";
    end GeneralTestBench;

    model Training2ndOrderScheme
      extends GeneralTestBench;
      SDEWES.Advection.SeccondOrderScheme solver(L = L, v = v) annotation(
        Placement(visible = true, transformation(origin = {34, 0}, extent = {{-22, -22}, {22, 22}}, rotation = 0)));
      Modelica.Blocks.Sources.TimeTable source(table=[0,0;2,0;2,1;5,1;5,-3;9,-7;12,3;17,3]) annotation(
        Placement(visible = true, transformation(origin = {-38, 1.77636e-15}, extent = {{-16, -16}, {16, 16}}, rotation = 0)));
    equation
      connect(source.y, solver.y_input) annotation(
        Line(points = {{-18, 0}, {12, 0}}, color = {0, 0, 127}, thickness = 1));
      annotation(
        experiment(StartTime = 0, StopTime = 50, Tolerance = 1e-6, Interval = 0.1));
    end Training2ndOrderScheme;

    model TrainingNeuralScheme
      extends GeneralTestBench;
      SDEWES.Advection.NeuralScheme solver(L = L, v = v) annotation(
        Placement(visible = true, transformation(origin = {40, 0}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
      Modelica.Blocks.Sources.TimeTable source(table=[0,0;2,0;2,1;5,1;5,-3;9,-7;12,3;17,3])   annotation(
        Placement(visible = true, transformation(origin = {-38, 1.77636e-15}, extent = {{-16, -16}, {16, 16}}, rotation = 0)));
    equation
      connect(source.y, solver.y_input) annotation(
        Line(points = {{-20, 0}, {20, 0}, {20, 0}, {20, 0}}, color = {0, 0, 127}));
    annotation(
        experiment(StartTime = 0, StopTime = 50, Tolerance = 1e-6, Interval = 0.1));
    end TrainingNeuralScheme;
    
  end Advection;

  package System_Identification
  end System_Identification;

  package InverseProblems
    connector Heat_port
      Real T;
      flow Real Q;
    end Heat_port;

    model TransientWall
      // FEM 1st order Galerkin 1D
      Heat_port heat_port_a;
      Heat_port heat_port_b;
      parameter Real T_init = 20;
      parameter Real L = 0.01;
      parameter Integer N = 20;
      parameter Real dx = L / N;
      parameter Real cp = 450;
      parameter Real k = 50;
      parameter Real rho = 7850;
      Real T[N - 1];
      Real Q[N];
      Real T_sensor = T[end] "#plot";
    initial equation
      heat_port_a.T = T_init;
      T = fill(T_init, N - 1);
      heat_port_b.T = T_init;
    equation
// Eq. for heat fluxes
      Q[1] = k * (heat_port_a.T - T[1]) / dx;
      Q[2:end - 1] = k * (T[1:end - 1] - T[2:end]) / dx;
      Q[end] = k * (T[end] - heat_port_b.T) / dx;
// ODEs for temperatures
      der(heat_port_a.T) * (dx / 2) * rho * cp = heat_port_a.Q - Q[1];
      der(T) * cp * rho * dx = Q[1:end - 1] - Q[2:end];
      der(heat_port_b.T) * (dx / 2) * rho * cp = Q[end] - heat_port_b.Q;
    end TransientWall;

    model NeumanBC
      parameter Real Qp[Nt](each min = -10000, each max = 10000) = fill(0, Nt) "#optimize";
      parameter Integer Nt = 40;
      parameter Real stopTime = 800;
      parameter Real t[Nt](each min = 0, each max = stopTime) = linspace(0, stopTime, Nt);
      Real Q = Modelica.Math.Vectors.interpolate(t, Qp, time);
      Heat_port heat_port;
    equation
      heat_port.Q = -Q;
    end NeumanBC;

    model NeumanBC_REF
      Modelica.Blocks.Sources.TimeTable data(table = [0, 0; 20, 0; 20, 3000; 100, 3000; 100, -3000; 150, -3000; 150, 0; 300, 1300; 300, 2000; 400, 2000; 400, -4000; 500, -4000; 500, 0; 600, 1000]);
      Heat_port heat_port;
      Real Q = data.y;
    equation
      heat_port.Q = -Q;
    end NeumanBC_REF;

    model RobinBC
      parameter Real Tamb = 20 "Ambient temperature";
      parameter Real alpha = 15 "Heat transfer coeffcionent";
      Heat_port heat_port;
    equation
      heat_port.Q = alpha * (heat_port.T - Tamb);
    end RobinBC;

    model Example1
      // reference simulation (represents data)
      NeumanBC_REF ref_boundary;
      TransientWall ref_wall;
      RobinBC ref_robinBC;
      //simulation that is being optimized
      NeumanBC opt_boundary;
      TransientWall opt_wall;
      RobinBC opt_robinBC;
      //loss
      Real loss = (ref_wall.T[end] - opt_wall.T[end]) ^ 2 "#objective";
    equation
//reference sumulation connections
      connect(ref_boundary.heat_port, ref_wall.heat_port_a);
      connect(ref_wall.heat_port_b, ref_robinBC.heat_port);
// optimized simulation conections
      connect(opt_boundary.heat_port, opt_wall.heat_port_a);
      connect(opt_wall.heat_port_b, opt_robinBC.heat_port);
      annotation(
        experiment(StopTime = 800, Interval = 4));
    end Example1;
  end InverseProblems;
  annotation(
    uses(Modelica(version = "3.2.2")));
end SDEWES;
