import pstats
p = pstats.Stats("profile.log")
 
p.strip_dirs().sort_stats(-1).print_stats()


with open('profile.txt','w') as stream:
    stats = pstats.Stats('profile.log', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()