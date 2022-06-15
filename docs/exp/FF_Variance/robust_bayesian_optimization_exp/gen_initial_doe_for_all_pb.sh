python gen_initial_xs.py  -n 5 -b 100 -pb SinLinear -tau 0.001 -kw_pb {}
python gen_initial_xs.py  -n 5 -b 100 -pb Forrester -tau 0.001 -kw_pb {}
python gen_initial_xs.py  -n 10 -b 100 -pb Branin -tau 0.01 -kw_pb {}
python gen_initial_xs.py  -n 10 -b 100 -pb Conceptual_Low_Drag_Wing_Design -tau 0.01 -kw_pb '{"weight": 100, "velocity": 20}'
python gen_initial_xs.py  -n 15 -b 100 -pb Hartmann3 -tau 0.01 -kw_pb {}
python gen_initial_xs.py  -n 15 -b 100 -pb Robot_Pushing_3D -tau 0.01 -kw_pb '{"obj_loc_x": 2, "obj_loc_y": 3}'
