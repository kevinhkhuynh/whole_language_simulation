# whole_language_simulation

Instructions

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

4. To run the code with a sample tableau, run the following command from the base directory of the repository. The [FILE].csv must be passed as an argument into the .jar file for the model to run.
   - java -jar target/whole_language_simulation-1.0-SNAPSHOT-jar-with-dependencies.jar example.csv 
   
5. To run your own tableau, it must follow a strict format to ensure correctness. This format was chosen to allow in hopes of generalizing the model to work for different problems.

  - The tableau must be in .csv format.
  - The first cell of the first line must be "Features" followed by the list of features.
  - The first cell of the next set of lines must be "Transition" followed by the name of the parent node, the name of the child node, and the list of features for the given transition. Optionally, cells of zero can be left as blank in a list of features for a given transition. However, the length of the list of features must equal to the number given in FEATURES.
  - The first cell of the last set of lines must be "Observation" followed by the name of the root node, the name of the leaf node, and the frequency of this given observation.
  - One thing to note is that no two states can be given the same name (state names are case-sensitive), even if they semantically mean the same thing. For example, if I would like to create a path for three different nodes [1:A -> 2:B -> 3:A], the MEMM would model this as [1:A -> 2:B -> 1:A]. Thus, any two states with the same name must be differentiated with a beginning marker (X) where x is the marker for the number of the state. In our example, the path would be marked as [1:A -> 2:B -> 3:(2)A], where (2) is the marker showing that this state is different from the previous. The model is programmed to automatically eliminate these markers after training.
  - Due to problems, any accented characters or irregular characters are recommended to be converted from (á to 'a, and à to \`a) to ensure correctness of the model. 
  - Lastly, as this is a .csv file, any cells that contain a comma will cause the model to crash. Please consider substituting another character for commas.
  - An example .csv file is given in whole_language_simulation/example.csv.

6. There are multiple model parameters that can be adjusted depending on your particular model by passing in optional parameters to the .jar file. The optional parameters can be inputted in any order. Also, keep in mind that passing in nonsensical values for some of these parameters may cause the model to crash.

  Optional Parameters
  - -len_trans [int] (default = 3) : The length of the word chain path in number of transitions
  - -len_states [int] (default = 4): The length of the word chain path in number of states
  - -threshold [double] (default = None):  Minimum log-Likelihood of random initialization necessary to begin the EM algorithm
  - -removemarkers [boolean] (default = true): Remove markers (for states that are numbered for differentiation)
  - -numtrials [int] (default: 50): The number of trials
  - -tolerance [double] (default: 0.25): The tolerance in which the EM algorithm is considered converged
  - -l2param [double] (default: 10): The l2 regularization penalty of the MEMM
  - -negparam [double] (default: 500): The penalty given to a negative weight of the MEMM
  - -stepsize [double] (default: 0.001): The step size of the Conjugate Gradient

  Example command with optional parameters:
  - java -jar target/whole_language_simulation-1.0-SNAPSHOT-jar-with-dependencies.jar example.csv -len_trans 4 -len_states 2 -threshold 300 -removemarkers false -numtrials 20 -tolerance 20 -l2param 200 -negparam 10 -stepsize 10
