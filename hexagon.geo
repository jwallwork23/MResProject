// Gmsh project created on Fri Feb 10 13:02:42 2017
//+
Point(1) = {-0.4, 0.5, -0, 1.0};
//+
Point(2) = {0.2, 0.5, 0, 1.0};
//+
Point(3) = {0.5, -0, 0, 1.0};
//+
Point(4) = {0.2, -0.5, 0, 1.0};
//+
Point(5) = {-0.4, -0.5, 0, 1.0};
//+
Point(6) = {-0.7, 0, -0, 1.0};
//+
Line(1) = {6, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {4, 5};
//+
Line(6) = {5, 6};
//+
Line Loop(7) = {1, 2, 3, 4, 5, 6};
//+
Plane Surface(8) = {7};
