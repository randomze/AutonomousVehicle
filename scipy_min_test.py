import numpy as np
import scipy.optimize as spopt

np.set_printoptions(precision=3)

E_budget = 10
M = 1
# lengths of paths
s = np.array([3,4, 5,1, 34, 12, 1, 2])
max_v_lims = np.array([10, 10, 1, 10, 10, 10, 10, 10])
# set to above 0 just to avoid divisions by 0 
min_v_lims = np.ones_like(max_v_lims)*1e-6
N = len(s)
vi = 0
vf = 0
# optimization is done in middle_v vector which is [v_1, ..., v_(N-1)],
# where v_i are the edge velocities in between each segment
def fun(m_v):
    v = np.block([vi, m_v, vf])

    avg_v = (v[:-1]+v[1:])/2
    t = np.divide(s, avg_v)
    f = t.sum()
    
    # print(f"\tv {v}\n\tavg {avg_v})\n\tf {f}")
    return f

def total_E_spent(m_v):
    v = np.block([vi, m_v, vf])

    v_sqr = v**2
    dif = (v_sqr[1:] - v_sqr[:-1])
    
    # print(dif)
    # braking does not recuperate energy
    dif = dif[dif>0]
    # print(dif)
    return M*dif.sum()/2

cons = ({'type': 'eq', 'fun': lambda m_v: total_E_spent(m_v) - E_budget})
# each velocity must respect the min and max limits of both its neighbor paths
bnds = [(max(min_v_lims[i], min_v_lims[i+1]), min(max_v_lims[i], max_v_lims[i+1])) for i in range(N-1)]


ini_v = np.ones(N-1)

sol = spopt.minimize(fun, ini_v, method='SLSQP', bounds=bnds,  constraints=cons)

print(sol)
v = np.block([vi, sol.x, vf])
print((f'With vi = {vi}, vf = {vf}, E = {E_budget}, max speed limits = {max_v_lims} '
        f'and M={M}, the obtained velocities are:\n\tv = {v}'))


max_v = np.sqrt(vi**2 + 2*E_budget/M)
print(f"Max possible velocity with this energy and mass would be {max_v}")
if vf > max_v:
    print(f"ERROR: vf = {vf} is bigger than max possible velocity, so expect no solution")
    
