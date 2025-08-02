 -- CompOptim --

-- OVERVIEW -------------------------------------------------------------------------------------

A PyTorch implementation of compressed optimization for decentralized training of neural networks 
using a variety of methods including SGD, Adam, and AdaGrad, all with several alterable parameters 
such as differing communication schemes and optional compression components.

-------------------------------------------------------------------------------------------------

-- DOCUMENTATION---------------------------------------------------------------------------------

Documentation for each function can be found within the "DOCS" subdirectory. This documentation 
contains explanations of the input for each function, what each input does, what the function does
, return types, and return formatting.

Additionally, some documentation can be found within the .py files themselves in the form of
comments. However these comments are not guaranteed to be up to date. (I suppose neither are the
DOCS files, but I will do my best.)

-------------------------------------------------------------------------------------------------

-- ORGANIZATION ---------------------------------------------------------------------------------

src/Optimizer :- A simple class for storing optimization schemes. It inherets from CommNet to set 
up its communications, and needs to be handed a model upon initialization. In general this model
is handed to it by a DistDataModel.
	1. Optimization functions.
	2. Includes Adam and AdamW.
	3. Includes AdaGrad.
	4. Includes AmsGrad.

src/CommNet :- A class which controls the network communications of our distributed models. 
This class contains the functionality to:
	1. compress data during communication
	2. (TODO) implement many predefined network topologies
	3. set communication schemes

src/DistDataModel :- The class that sets up our data distributed models on a network. The class 
itself sets up our model, and defines an instance of an Optimizer within itself that defines the 
network communications. This class contains the functionality to:
	1. Easily define models on a given network topology.
	2. Automatically distribute training sets over that topology.
	3. Perform training with various optimizers from the "Optimizer" class, using custom 
	communication schemes.
	4. Track results and save them.

(TODO) src/DistParamModel :- This class is focused on distributing the parameters of models over 
a network. This class inherets from "CommNet", and has the funcitonality to:
	1. (TODO) Define models that are distributed over a given network topology.
	2. (TODO) Perform training with various topologies.
	3. (TODO) Define heuristically decided splits for models that match with network topology.
	4. (TODO) Track results and save them.

src/Tracker :- A class that deals with tracking and evaluating models. Is used inside of 
instances of DistParamModel to report loss and accuracy among other things. Tacks onto an instance
of the "Optimizer" class, and uses its instance of COMM_WORLD for communication. 
Contains functions to:
	1. Track accuracy and loss.
	2. Report information to console.
	2. Output data from each rank to a yaml file.
	3. Gather data from all ranks and output to a single yaml file. 

src/Visualizer :- A class for plotting results in various styles. The visualizer is 
loaded with data output from a DistDataModel, or a DIstParamModel, and has functions for:
	1. Loading data from the 'results' folder.
	2. Creating a new folder within 'figs'.
	3. Visualizing data in a variety of ways.

-------------------------------------------------------------------------------------------------

-- RUNNING --------------------------------------------------------------------------------------

* Currently, you can run a simple instance via the "experiment.py" file by trying:
	>> mpirun -np 4 python3 experiment.py <<
	This will train LeNet5 on 4 MPI ranks on the FMNIST dataset for some number of epochs.
	
-------------------------------------------------------------------------------------------------

-- KNOWN BUGS / WEIRDNESS -----------------------------------------------------------------------

 * Adam in DistDataModel.train() does not work reliably if x is communicated before x is updated.
 	>> This is likely an error since this results in the same method as Chen 2022.

 * Currently the network needs to have an even number of nodes. E.G. self.nprocs within CommNet 
 should be such that self.nprocs%2 == 0. 
 	
-------------------------------------------------------------------------------------------------
