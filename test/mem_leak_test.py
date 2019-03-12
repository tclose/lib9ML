from itertools import chain
from pympler import muppy, summary
from nineml.utils.testing.comprehensive_example import instances_of_all_types

all_objects = muppy.get_objects()


for instance in chain(*(d.values() for d in instances_of_all_types.values())):
    clone = instance.clone()
    del clone
    new_all_objects = muppy.get_objects()
    if len(new_all_objects) != len(all_objects):
        s = summary.summarize(all_objects)
        print(instance)
        summary.print_(s)
        all_objects = new_all_objects
