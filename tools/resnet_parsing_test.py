import memory_util


#memory_util.print_memory_timeline(log, ignore_less_than_bytes=0)
memory_util.plot_memory_timeline(open('output.log').read())
