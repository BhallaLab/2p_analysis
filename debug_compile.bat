c++ -O3 -g -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -I. -I/home/bhalla/homework/HILLTAU/REPO/HillTau/extern/pybind11/include analyze2p.cpp bind2p.cpp -o a2p$(python3-config --extension-suffix)
