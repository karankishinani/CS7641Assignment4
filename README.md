# CS7641 Assignment 4

### Markov Decision Processes

**GitHub Link:** https://github.gatech.edu/kkishinani3/CS7641Assignment4

## References
- BURLAP: http://burlap.cs.brown.edu/
- Base code: https://github.com/JonathanTay/CS-7641-assignment-4

## PROGRAMING LANGUAGE AND DEPENDENCIES

This repository uses the BURLAP Library to implement the Value Iteration, Policy Iteration, and Q-Learning algorithms. 

### Problem 1: Slippery World Treasure Hunt
	- `easyGW.py`

### Problem 2: Battery Recharge Maze
	- `hardGW.py`

_Note_: All these code files above have been written in Jython and Python 3. Please refer to the Jython section below for information on how to run each one.

Make sure to have the following libraries installed. If not, you can install using ```pip install <library name>```

### Libraries used:
- matplotlib
- pandas
- numpy
- pickle

### Jython and Running the Problems

1. Please make sure you have Jython 2.7.1 installed in your machine. If you have Homebrow installed in your machine, run the following command to install the latest version of Jython:

    `brew install jython`

2. Each of the experiment files for the neural network and optimization toy problems are writting using Jython. To run each individual experiment run the following from the home directory:

    '''
    jython -J-Xms6000m -J-Xmx6000m easyGW.py 
	jython -J-Xms6000m -J-Xmx6000m hardGW.py
	'''

### Generating Plots

The code for generating plots is contained in the plotting files written in Python 3. Run the following command to generate the plots which will be stored in the `./out/plots` directory.

	```
	python3 ploteasy.py
	python3 plothard.py
	```