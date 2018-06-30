# whole_language_simulation

This repository uses Factorie to implement a Maximum Entropy Markov Model primarily for use in the domain of phonology and linguistics. Given a tableau consisting of many hidden states and a handful of observations, the MEMM calculates the weights of the constraints.

Instructions to Run the Code

1. Download the whole_language_simulation repository from GitHub.

2. Install Java JDK 8.
  - http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html

3. Enter the base directory of the repository.

  For Mac and Linux Users:
  - Open up Terminal
  - cd FULL_PATH_OF_BASE_DIRECTORY_OF_REPOSITORY
  
  For Windows Users:
  - Open up the Windows Command Prompt.
  - https://introcs.cs.princeton.edu/java/15inout/windows-cmd.html

4. To run the code with a sample tableau, run the following command from the base directory of the repository. The model can currently only train on one tableau at a time. Thus, if there are multiple tableaus, they must be merged into a single file. The [FILE].csv must be passed as an argument into the .jar file for the model to run.
   - java -jar target/whole_language_simulation-1.0-SNAPSHOT-jar-with-dependencies.jar tongan.csv
   
5. After the model finishes running, it outputs the trial run with the best negative log-likelihood along with the features and their given weights. It additionally outputs the probabilities and frequencies of the ground truth observations along with the model's predicted probabilities. The outputted file will have the path: whole_language_simulation/MaximumEntropyMarkovModel.txt. An example output file is given in whole_language_simulation/tongan.txt.
   
5. To run your own tableau, it must follow a strict format to ensure correctness. This format was chosen to allow in hopes of generalizing the model to work for different problems.

  - The tableau must be in .csv format.
  - The first cell of the first line must be "Features" followed by the list of features.
  - The first cell of the next set of lines must be "Transition" followed by the name of the parent node, the name of the child node, and the list of features for the given transition. Optionally, cells of zero can be left as blank in a list of features for a given transition. However, the length of the list of features must equal to the number given in FEATURES.
  - These "Transition" lines must be grouped by word chain. The first "Transition" line of each new word chain should be marked as "*Transition". The parent node in this "*Transition" line will be marked as the root node of the word chain. 
  - The first cell of the last set of lines must be "Observation" followed by the name of the root node, the name of the leaf node, and the frequency of this given observation.
  - One thing to note is that no two states can be given the same name (state names are case-sensitive), even if they semantically mean the same thing. For example, if I would like to create a path for three different nodes [1:A -> 2:B -> 3:A], the MEMM would model this as [1:A -> 2:B -> 1:A]. This is because if two states have the same name, the model has no way of knowing which one of the two states should be connected by a transition. Thus, any two states with the same name must be differentiated with a beginning marker (X) where x is the marker for the number of the state. In our example, the path would be marked as [1:A -> 2:B -> 3:(2)A], where (2) is the marker showing that this state is different from the previous. The model is programmed to automatically eliminate these markers after training.
  - Due to problems, any accented characters or irregular characters are recommended to be converted from (á to 'a, and à to \`a) to ensure correctness of the model. 
  - Lastly, as this is a .csv file, any cells that contain a comma will cause the model to crash. Please consider substituting another character for commas.
  - An example .csv file is given in whole_language_simulation/tongan.csv.

7. There are multiple model parameters that can be adjusted depending on your particular model by passing in optional parameters to the .jar file. The optional parameters can be inputted in any order. Also, keep in mind that passing in certain values for some of these parameters may cause the model to never converge.

  Optional Parameters
  - -threshold [double] (default = None):  Minimum log-Likelihood of random initialization necessary to begin the EM algorithm
  - -removemarkers [boolean] (default = true): Remove markers (for states that are numbered for differentiation)
  - -numtrials [int] (default: 50): The number of trials
  - -tolerance [double] (default: 0.25): The tolerance in which the EM algorithm is considered converged
  - -l2param [double] (default: 10): The l2 regularization penalty of the MEMM
  - -negparam [double] (default: 500): The penalty given to a negative weight of the MEMM
  - -stepsize [double] (default: 0.001): The step size of the Conjugate Gradient

  Example command with optional parameters:
  - java -jar target/whole_language_simulation-1.0-SNAPSHOT-jar-with-dependencies.jar tongan.csv -threshold -700 -removemarkers false -numtrials 20 -tolerance 0.2 -l2param 15 -negparam 600 -stepsize 0.0001
  
  
Instructions to Alter the Code
1. Complete steps 1-3 from the previous instructions.
1. Download Apache Maven
  - https://maven.apache.org/download.cgi
2. Alter the source code.
3. In the base directory of the project, run the following command:
  - mvn package
4. The new source code will be built and the model can now be run with the following command:
  - java -jar target/whole_language_simulation-1.0-SNAPSHOT-jar-with-dependencies.jar [FILE].csv

# graph_viz_printer

An automatic GraphViz printer can be used to print the transition structures with the optimized probabilities.

Instructions to Run the Code

1. Download and install GraphViz: https://www.graphviz.org/download/

2. Enter the base directory of the repository.

  For Mac and Linux Users:
  - Open up Terminal
  - cd FULL_PATH_OF_BASE_DIRECTORY_OF_REPOSITORY
  
  For Windows Users:
  - Open up the Windows Command Prompt.
  - https://introcs.cs.princeton.edu/java/15inout/windows-cmd.html

3. Run graph_viz_printer with a tableau and a corresponding outputted results file.

   Example command with optional parameters:
  - java -jar graph_viz_printer.jar antilla.csv antilla.txt

4. The corresponding dot files will be printed into the directory dotFiles. A dot file corresponds to one word chain and can be interpreted by GraphViz to obtain a .png file.

5. Enter the dotFiles directory of the repository.

6. To obtain the .png files, for each .dot file, run the following command in the directory dotFiles. The command creates a corresponding .png file for each dot file.
  - dot [FILE].dot -Tpng -o [FILE (Name can differ)].png
  
7. Using the .png file as a guide, the labels inside the nodes of the MEMM can be changed by altering the labels within the .dot file and repeating step 4. For example, we can alter the line "X1[label=<MELITIANE>]" within the .dot file to "X1[label=<[(mèli)ti(áne)]>]".
