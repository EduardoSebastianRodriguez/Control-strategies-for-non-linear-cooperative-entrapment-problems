# Control strategies for non linear cooperative entrapment problems

This repository stores and shares the software implementation of my End-of-Degree project. The structure is detailed in the following lines:

* "functions.py" contains the core of the project. It implements the two main classes of the project, Prey and Hunter, along with the different dynamical models and numerical functions. It also includes a set up functions which implements the simulation of an specific configuration of preys, hunters and models. Finally, some auxiliary functions can be found.

* "main.py" contains the top layer of the project, implementing the previous actions needed to run a complete simulation. The numerical solver for the initial operating point is included here. Moreover, the design of the controllers as a functions of the number of hunters, preys and their corresponding dynamical behavior is implemented in this file. In essence, the main script implements the complete algorithm.

* "control" includes some basic control functions neccesary to run the other files.

* "test" includes three files: two of them implement Monte-Carlo experiments to assess the performance of the different control strategies. The "reload_data.py" file aims to load data previously obtained with the experiments in order to display figures again or post-proccess the data.

* "benchmark" includes some extra files created to prototipe the different controllers. It also includes the implementation of 1 vs 1 models, derivated from the four main ones.
