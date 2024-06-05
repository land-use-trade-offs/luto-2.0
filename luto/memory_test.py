import sys
sys.path.append('.')

from memory_profiler import profile
import luto.simulation as sim  
from luto.tools import write

data = sim.load_data()
sim.run(data, 2010, 2030)
write.write_outputs(sim)
