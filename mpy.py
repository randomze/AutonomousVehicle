import matplotlib.pyplot as plt
import numpy as np
import matlab.engine

eng = matlab.engine.start_matlab()

eng.workspace['gain'] = 100

simout = eng.sim('test')

gts = eng.get(simout, 'gain_times_sin') # returns timeseries matlab object

gts_data = eng.get(gts, 'Data') # returns data

gts_time = eng.get(gts, 'Time') # returns time

plt.plot(gts_time, gts_data)
plt.show()
