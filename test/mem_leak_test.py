import resource
from nineml.utils.testing.comprehensive_example import instances_of_all_types

multi_instances = instances_of_all_types['MultiDynamicsProperties']


def print_mem(msg):
    print('Memory {}: {} (kb)'.format(
        msg, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


for instance in multi_instances.values():
    print_mem('before clone of {}'.format(instance))
    clone = instance.clone()
    print_mem('after clone of {}'.format(instance))
    del clone
    print_mem('after deletion of {}'.format(instance))
