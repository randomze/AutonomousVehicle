import matplotlib.pyplot as plt
import matlab.engine

opt = "-desktop"

cur_engs = matlab.engine.find_matlab()
if len(cur_engs) == 0:
    eng = matlab.engine.start_matlab(option=opt)
else:
    eng = matlab.engine.connect_matlab(name=cur_engs[0])

eng.workspace['gain'] = 100

eng.eval("other_gain = 200;", nargout=0)

simout = eng.sim('test')

gts = eng.get(simout, 'gain_times_sin') # returns timeseries matlab object

gts_data = eng.get(gts, 'Data') # returns data

gts_time = eng.get(gts, 'Time') # returns time

plt.plot(gts_time, gts_data)
plt.show()
