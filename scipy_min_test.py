import numpy as np
import scipy.optimize as spopt

np.set_printoptions(precision=3)
np.random.seed(23)

E_budget = 10e3
mass = 1000
N = 30
idle_power = 1
# lengths of paths
s = np.random.uniform(1, 40, (N,))
max_v_lims = np.random.uniform(1, 10, (N,))
# set to above 0 just to avoid divisions by 0
min_v_lims = np.ones_like(max_v_lims)*1e-6
vi = 0
# optimization is done in middle_v vector which is [v_1, ..., v_(N-1)],
# where v_i are the edge velocities in between each segment

cost_scale = 256


def travel_time(path_velocities):  # Optimization cost, total time of travel
    path_travel_times = np.divide(s, path_velocities)
    return path_travel_times.sum()/cost_scale


def jac(path_velocities):  # jacobian of travel time
    return - np.divide(s, path_velocities**2)/cost_scale


def total_E_spent(path_velocities):
    v = np.block([vi, path_velocities])
    v_sqr = v**2
    diff = (v_sqr[1:] - v_sqr[:-1])
    # braking does not recuperate energy
    diff = diff[diff > 0]
    return mass*diff.sum()/2 + travel_time(path_velocities)*idle_power*cost_scale


cons = ({'type': 'eq', 'fun': lambda m_v: E_budget - total_E_spent(m_v)})
# each velocity must respect the min and max limits of both its neighbor paths
bnds = list(zip(min_v_lims, max_v_lims))


ini_v = np.ones(N)

sol = spopt.minimize(travel_time, ini_v, method='SLSQP', jac=jac, bounds=bnds,
                     constraints=cons, options={"maxiter": 3000})

print(sol)
v = np.block([vi, sol.x])
# print((f'With vi = {vi}, E = {E_budget}, max speed limits = {max_v_lims} '
#    f'and M={mass}, the obtained velocities are:\n\tv = {v}'))


max_v = np.sqrt(vi**2 + 2*E_budget/mass)
print(f"Max possible velocity with this energy and mass would be {max_v}")
