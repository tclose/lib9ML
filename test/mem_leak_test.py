from itertools import chain
from pympler import muppy, summary
import psutil
from nineml.utils.testing.comprehensive_example import instances_of_all_types
import os

# Get reference to current process
process = psutil.Process(os.getpid())

num_objects = len(muppy.get_objects())
mem_usage = process.memory_info()[0]


for instance in chain(*(d.values() for d in instances_of_all_types.values())):
    clone = instance.clone(clone_definitions=False,
                           validate=False)
    del clone
    new_mem_usage = process.memory_info()[0]
    if new_mem_usage != mem_usage:
        all_objects = muppy.get_objects()
        if len(all_objects) != num_objects:
            num_objects = len(all_objects)
            s = summary.summarize(all_objects)
            print('{} ({})'.format(instance, type(instance)))
            summary.print_(s)
            del s
        del all_objects
        mem_usage = new_mem_usage
