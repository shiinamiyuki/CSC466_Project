import os
import subprocess

test_set = [
    'beast',
    'chimpanzee-hand',
    'ikea-lamp',
    'knight',
    'robot-arm'
]
tols = {
    'beast':(0.01, 0.002),
    'chimpanzee-hand':( 0.0001, 0.0002),
    'ikea-lamp':(0.0001, 0.0002),
    'knight':( 0.0001, 0.0002),
    'robot-arm':( 0.0001, 0.0002),
}
procs = []
for test in test_set:
    n = 1000
    for solver in ['gd','bfgs','gauss']:
        for mode in ['i','u']:
            if solver == 'gd':
                n = 100
            else:
                n = 1000
            out = 'test-result-{}-s-{}-n-{}-m-{}.txt'.format(test,solver, n, mode)
            cmd = '../data/{}.json -s {} -n {} -{} -o {} --tol {} {}'.format(test, solver, n, mode, out, tols[test][0], tols[test][1])
            cmd = cmd.split(' ')
            print(cmd)
            procs.append(subprocess.Popen(['./random_test'] + cmd))


for p in procs:
    p.wait()
