import matlab.engine
import matlab
import matlab_utils.utils as m_utils
import environment.road as road

eng: matlab.engine.MatlabEngine = m_utils.get_engine()

sim_initial_conditions = {
    'x_0': 0.0,
    'y_0': 0.0,
    'theta_0': 0.0,
    'phi_0': 0.0,
    'v_0': 0.0,
}

car_description = {
    'L': 2.2,
    'Lr': 0.566,
    'Lf': 0.566,
    'd': 0.64,
    'r': 0.256,
    'Length': 3.332,
    'Width': 1.508,
    'M': 800.0,
}

r_name = 'road_img'
lat, long = (38.7367256,-9.1388871)
zoom = 16
road_img, road_graph = road.get_road_info((lat, long), zoom, single_channel=True)

road_description = {
    'meters_per_pixel': float(road.zoom_to_scale(zoom, lat)),
}

m_utils.to_workspace(sim_initial_conditions, car_description, road_description)
m_utils.big_var_to_workspace(r_name, road_img)
eng.eval(f'{r_name} = logical({r_name});', nargout=0)
simout = eng.sim('simulation')


#cv2.imshow('road_img', road_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()