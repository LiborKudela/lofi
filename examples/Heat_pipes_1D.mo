package Heat_pipes_1D
  connector Heat_port
    flow Modelica.SIunits.HeatFlowRate Q;
    Modelica.SIunits.Temperature T;
    annotation(
      Icon(graphics = {Ellipse(fillColor = {239, 41, 41}, fillPattern = FillPattern.Solid, lineThickness = 0.75, extent = {{-100, 100}, {100, -100}}, endAngle = 360)}));
  end Heat_port;

  model Node
    parameter Modelica.SIunits.HeatCapacity C(min=-10, max=10) = 1.0 "#optimize";
    Heat_pipes_1D.Heat_port heat_port annotation(
      Placement(visible = true, transformation(origin = {0, 0}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-8.88178e-15, 2.13163e-14}, extent = {{-50, -50}, {50, 50}}, rotation = 0)));
  equation
    der(heat_port.T)*C = heat_port.Q; 
  annotation(
      Icon(graphics = {Line(origin = {-28.3194, 26.6896}, points = {{3, 5}, {-3, -5}}), Text(origin = {-78, 22}, rotation = 90, extent = {{-102, 12}, {58, -28}}, textString = "%name")}, coordinateSystem(initialScale = 0.1)));
  end Node;

  model Resistance
    parameter Modelica.SIunits.ThermalResistance R(min=-10, max=10) = 1.0 "#optimize";
    Heat_pipes_1D.Heat_port heat_port_a annotation(
      Placement(visible = true, transformation(origin = {-90, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {-90, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Heat_pipes_1D.Heat_port heat_port_b annotation(
      Placement(visible = true, transformation(origin = {90, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {90, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  equation
    heat_port_a.Q = -heat_port_b.Q;
    heat_port_a.Q*R = (heat_port_a.T - heat_port_b.T);
  annotation(
      Icon(graphics = {Rectangle(fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid, lineThickness = 0.75, extent = {{-80, 20}, {80, -20}}), Text(origin = {2, -44}, rotation = 180, extent = {{-80, 22}, {80, -16}}, textString = "%name")}, coordinateSystem(initialScale = 0.1)));
  end Resistance;

  model Outer_interaction
    parameter Modelica.SIunits.CoefficientOfHeatTransfer alpha = 100.0;
    parameter Real scale_coefficient = 1.0 "#optimize";
    Heat_pipes_1D.Heat_port heat_port annotation(
      Placement(visible = true, transformation(origin = {90, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0), iconTransformation(origin = {90, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Modelica.Blocks.Interfaces.RealInput T_water annotation(
      Placement(visible = true, transformation(origin = {-100, 0}, extent = {{-20, -20}, {20, 20}}, rotation = 0), iconTransformation(origin = {-80, -2}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
  equation
    heat_port.Q = scale_coefficient * alpha * (heat_port.T-T_water);
  annotation(Icon(graphics = {Rectangle(fillColor = {255, 255, 255}, fillPattern = FillPattern.Solid, lineThickness = 0.75, extent = {{-80, 80}, {80, -80}}, radius = 10), Text(origin = {4, -1}, extent = {{-56, 23}, {56, -23}}, textString = "%name")}, coordinateSystem(initialScale = 0.1)));
  end Outer_interaction;

  model Example
  Heat_pipes_1D.Node node annotation(
      Placement(visible = true, transformation(origin = {0, 120}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Node node1 annotation(
      Placement(visible = true, transformation(origin = {-130, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Resistance resistance annotation(
      Placement(visible = true, transformation(origin = {-130, 90}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Heat_pipes_1D.Resistance resistance1 annotation(
      Placement(visible = true, transformation(origin = {-130, 30}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Heat_pipes_1D.Node node2 annotation(
      Placement(visible = true, transformation(origin = {-130, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Resistance resistance2 annotation(
      Placement(visible = true, transformation(origin = {-130, -30}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Heat_pipes_1D.Node node3 annotation(
      Placement(visible = true, transformation(origin = {-130, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Resistance resistance3 annotation(
      Placement(visible = true, transformation(origin = {-130,-90}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Heat_pipes_1D.Node node4 annotation(
      Placement(visible = true, transformation(origin = {-130, -120}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Node node5 annotation(
      Placement(visible = true, transformation(origin = {130, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Resistance resistance4 annotation(
      Placement(visible = true, transformation(origin = {130, 30}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Heat_pipes_1D.Resistance resistance5 annotation(
      Placement(visible = true, transformation(origin = {130, 90}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Heat_pipes_1D.Node node6 annotation(
      Placement(visible = true, transformation(origin = {130, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Resistance resistance6 annotation(
      Placement(visible = true, transformation(origin = {130, -30}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Heat_pipes_1D.Node node8 annotation(
      Placement(visible = true, transformation(origin = {130, -60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Resistance resistance7 annotation(
      Placement(visible = true, transformation(origin = {130, -90}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Heat_pipes_1D.Node node9 annotation(
      Placement(visible = true, transformation(origin = {130, -120}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Resistance resistance8 annotation(
      Placement(visible = true, transformation(origin = {-96, -120}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Resistance resistance9 annotation(
      Placement(visible = true, transformation(origin = {-30, -120}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Resistance resistance10 annotation(
      Placement(visible = true, transformation(origin = {30, -120}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Node node7 annotation(
      Placement(visible = true, transformation(origin = {-60, -120}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Node node10 annotation(
      Placement(visible = true, transformation(origin = {0, -120}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Node node11 annotation(
      Placement(visible = true, transformation(origin = {60, -120}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Resistance resistance12 annotation(
      Placement(visible = true, transformation(origin = {96, -120}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Outer_interaction SUPPLY annotation(
      Placement(visible = true, transformation(origin = {-170, -120}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Heat_pipes_1D.Outer_interaction RETURN annotation(
      Placement(visible = true, transformation(origin = {170, -120}, extent = {{10, -10}, {-10, 10}}, rotation = 0)));
  Heat_pipes_1D.Outer_interaction AIR annotation(
      Placement(visible = true, transformation(origin = {0, 150}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  equation
    connect(node4.heat_port, resistance3.heat_port_b) annotation(
      Line(points = {{-130, -120}, {-130, -99}}, thickness = 0.75));
    connect(resistance5.heat_port_b, node6.heat_port) annotation(
      Line(points = {{130, 81}, {130, 60}}, thickness = 0.75));
    connect(node6.heat_port, resistance4.heat_port_a) annotation(
      Line(points = {{130, 60}, {130, 39}}, thickness = 0.75));
    connect(resistance4.heat_port_b, node5.heat_port) annotation(
      Line(points = {{130, 21}, {130, 0}}, thickness = 0.75));
    connect(node5.heat_port, resistance6.heat_port_a) annotation(
      Line(points = {{130, 0}, {130, -21}}, thickness = 0.75));
    connect(resistance6.heat_port_b, node8.heat_port) annotation(
      Line(points = {{130, -39}, {130, -60}}, thickness = 0.75));
    connect(node8.heat_port, resistance7.heat_port_a) annotation(
      Line(points = {{130, -60}, {130, -81}}, thickness = 0.75));
    connect(resistance7.heat_port_b, node9.heat_port) annotation(
      Line(points = {{130, -99}, {130, -120}}, thickness = 0.75));
    connect(resistance5.heat_port_a, node.heat_port) annotation(
      Line(points = {{130, 99}, {130, 120}, {0, 120}}, thickness = 0.75));
    connect(node.heat_port, resistance.heat_port_a) annotation(
      Line(points = {{0, 120}, {-130, 120}, {-130, 99}}, thickness = 0.75));
    connect(resistance.heat_port_b, node1.heat_port) annotation(
      Line(points = {{-130, 80}, {-130, 80}, {-130, 60}, {-130, 60}}, thickness = 0.75));
    connect(node1.heat_port, resistance1.heat_port_a) annotation(
      Line(points = {{-130, 60}, {-130, 60}, {-130, 40}, {-130, 40}}, thickness = 0.75));
    connect(resistance1.heat_port_b, node2.heat_port) annotation(
      Line(points = {{-130, 22}, {-130, 22}, {-130, 0}, {-130, 0}}, thickness = 0.75));
    connect(node2.heat_port, resistance2.heat_port_a) annotation(
      Line(points = {{-130, 0}, {-130, 0}, {-130, -20}, {-130, -20}}, thickness = 0.75));
    connect(resistance2.heat_port_b, node3.heat_port) annotation(
      Line(points = {{-130, -38}, {-130, -38}, {-130, -60}, {-130, -60}}, thickness = 0.75));
    connect(node3.heat_port, resistance3.heat_port_a) annotation(
      Line(points = {{-130, -60}, {-130, -60}, {-130, -80}, {-130, -80}}, thickness = 0.75));
    connect(node4.heat_port, resistance8.heat_port_a) annotation(
      Line(points = {{-130, -120}, {-105, -120}}, thickness = 0.75));
    connect(resistance8.heat_port_b, node7.heat_port) annotation(
      Line(points = {{-87, -120}, {-60, -120}}, thickness = 0.75));
    connect(node10.heat_port, resistance9.heat_port_b) annotation(
      Line(points = {{0, -120}, {-21, -120}}, thickness = 0.75));
    connect(resistance9.heat_port_a, node7.heat_port) annotation(
      Line(points = {{-39, -120}, {-60, -120}}, thickness = 0.75));
    connect(resistance10.heat_port_a, node10.heat_port) annotation(
      Line(points = {{21, -120}, {0, -120}}, thickness = 0.75));
    connect(resistance10.heat_port_b, node11.heat_port) annotation(
      Line(points = {{39, -120}, {60, -120}}, thickness = 0.75));
    connect(node11.heat_port, resistance12.heat_port_a) annotation(
      Line(points = {{60, -120}, {87, -120}}, thickness = 0.75));
    connect(node9.heat_port, resistance12.heat_port_b) annotation(
      Line(points = {{130, -120}, {105, -120}}, thickness = 0.75));
    connect(SUPPLY.heat_port, node4.heat_port) annotation(
      Line(points = {{-160, -120}, {-130, -120}, {-130, -120}, {-130, -120}}, thickness = 0.75));
    connect(RETURN.heat_port, node9.heat_port) annotation(
      Line(points = {{160, -120}, {134, -120}, {134, -120}, {130, -120}}, thickness = 0.75));
    connect(AIR.heat_port, node.heat_port) annotation(
      Line(points = {{0, 140}, {0, 140}, {0, 120}, {0, 120}}, thickness = 0.75));
    annotation(
      Diagram(coordinateSystem(extent = {{-200, -200}, {200, 200}})),
      Icon(coordinateSystem(extent = {{-200, -200}, {200, 200}})));
  end Example;
  annotation(
    uses(Modelica(version = "3.2.2")));
end Heat_pipes_1D;