all_algs.py serves as the engine for all the experiments done in the project. 
Main parameters -

alg_func - choice of algorithm function to test.
N - number of items.
k - number of defective items.
phi, theta - error parameters. For the senesitivity test, I added a wrapper code that iterates over their values. 

test_regime 0 - Bernoulli. 1 - Regular. 2 - Double regular.


Some choices of parameters in the algorithm implementation:
- The scaling parameter for the perturbations cost in LP is set to 100. This however is too slow when the number of tests is 250, and so for these setting I set it at 10 * 1000.
- The filter size for the first round of filtering in the NDD algorithm is set to 3*k. 

These choices are based on running a few cases and choosing a reasonably performing parameter (They are not cherry picked!). 

The wolfram Mathematica python wrapper is required for the run of the linear program of LP. If you don't have this dependency (which requires a Mathematica license), you can remove the import and run any other algorithm. 
