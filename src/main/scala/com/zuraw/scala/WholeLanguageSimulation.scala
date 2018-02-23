package com.zuraw.scala

import cc.factorie.model._
import cc.factorie.la._
import cc.factorie.optimize._
import scala.math._
import scala.collection.JavaConverters._
import java.io._

/**
  * Created by kevinhuynh on 1/18/18.
  */
object WholeLanguageSimulation extends App {

  // Check that tableau was inputted and the number of parameters are correct
  if (args.length < 1 || args.length % 2 == 0) {
    System.err.println("Usage: whole_language_simulation [FILE].csv -optionalparameters ... \n Optional Parameters: -len_states [int] -threshold [double] -removemarkers [bool] -numtrials [int] -tolerance [double] -l2param [double] -negparam [double] -stepsize[double]")
  }

  // Create a random variable for Gaussian noise
  implicit val random = new scala.util.Random()

  // Name of tableau file
  var tableauFileName = args(0)

  // Check that tableau is of .csv format
  if (tableauFileName.takeRight(4) != ".csv") {
    System.err.println("Tableau must be in .csv format")
  }

  // The length of the word chain path in number of states. Pick the number of states that occur in the longest possible word chain path.
  var T_states = 4

  // Only continue with random initialization if log-likelihood is better than randomThreshold
  var allowRandomThreshold = false

  // Remove markers (for states that are numbered for differentiation)
  var removeMarkers = true

  // Minimum log-Likelihood of random initialization necessary to begin EM
  var randomThreshold = -600.0

  // Run numTrials amount of trials
  var numTrials = 50

  // Store learned constraint weights and log-likelihood for a given trial
  var trials = scala.collection.mutable.ArrayBuffer[Tuple2[DenseTensor1, Double]]()

  // Choose tolerance to end EM algorithm (experimentally derived)
  var tolerance = 0.25

  // The gradient step size
  var stepSize = 0.001

  // Define model
  val model = new MaximumEntropyMarkovModel

  // Obtain command line options
  for (arg <- args.drop(1).grouped(2).toList) {
    try {
      arg(0) match {
        case "-len_states" => T_states = arg(1).toInt
        case "-threshold" => allowRandomThreshold = true
          randomThreshold = arg(1).toDouble
        case "-removemarkers" => removeMarkers = arg(1).toBoolean
        case "-numtrials" => numTrials = arg(1).toInt
        case "-tolerance" => tolerance = arg(1).toDouble
        case "-l2param" => model.setRegularizationWeight(arg(1).toDouble)
        case "-negparam" => model.setNegativePenalization(arg(1).toDouble)
        case "-stepsize" => stepSize = arg(1).toDouble
      }
    }

    // Error checking for the formatting of the optional parameters
    catch {
      case _ : Throwable => System.err.println("The formatting of one or more of the optional parameters are incorrect")
        System.exit(1)
    }
  }

  // The length of the word chain path in number of transitions.
  val T_transitions = T_states - 1

  // Define states
  val States = scala.collection.mutable.HashMap[String, State]()

  // Read tableau file
  var tableau = readFile.read(tableauFileName).asScala

  // Variable to store features (the constraints)
  val features = scala.collection.mutable.ArrayBuffer[String]()

  // Hold training data that consists of [perceivedForm, spokenForm, count of spokenForms in data]
  var trainingData = scala.collection.mutable.ArrayBuffer[Tuple3[String, String, Double]]()

  // Hold current subset of variables, or word chain
  var subset = scala.collection.mutable.Set[String]()

  // Variable holding perceived form so that states for different word chains can be kept separate
  var perceivedForm = None:Option[String]

  // Define states, transitions, and candidate weights, and then add them to the model
  for ((line, linenumber) <- tableau.zipWithIndex) {

    // Split line into cells
    val cells = line.mkString(",").split(",")
    cells(0) match {

      // If reading "Features" line
      case "Features" =>

        // Check that features are only defined once
        if (features.nonEmpty) {
          System.err.println("The constraints have been redefined on line: " + linenumber + ". Only one definition for a set of constraints are allowed." )
          System.exit(1)
        }

        // Define constraints
        cells.drop(1).foreach { feature => features.append(feature) }

      // If reading "Transition" line
      case x if x == "*Transition" || x == "Transition" =>

        // Check that features are defined before any transitions are read from the tableau
        if (features.isEmpty) {
          System.err.println("Please define constraints before defining transitions." )
          System.exit(1)
        }

        // Start a new word chain
        if (cells(0) == "*Transition") {
          if (perceivedForm.isDefined && perceivedForm.get != cells(1)) {
            model.addSubset(perceivedForm.get, subset)
            subset = subset.empty
          }
          perceivedForm = Some(cells(1))
        }

        // Parse through tableau to obtain X, Y (X->Y) and feature values
        val candidateWeights = new DenseTensor1(features.size, 0.0)
        for ((element, index) <- cells.zipWithIndex) {
          index match {

            // Ignore "Transition" cell
            case 0 =>

            // Add X and Y to list of variables and to current word chain
            case 1 => if (States.get(cells(index)).isEmpty) {
              States.put(cells(index), new State(cells(index)))
              subset.add(cells(index))
            }
            case 2 => if (States.get(cells(index)).isEmpty) {
              States.put(cells(index), new State(cells(index)))
              subset.add(cells(index))
            }

            // Parse the features of the tableau
            case _ =>

              // Check that the number of features in the transition do not go over the number of actual features
              if (index - 3 >= features.size) {
                System.err.println("There are more features than there are defined constraints on line: " + linenumber)
                System.exit(1)
              }

              // Put tableau feature into weights
              element match {
                case "" => candidateWeights(index - 3) = 0
                case nonBlank => candidateWeights(index - 3) = nonBlank.toDouble
              }
          }
        }

        // Add transition to model
        model += new MarkovTransition(States(cells(2)), States(cells(1)), candidateWeights)

      // If reading "Observation" line
      case "Observation" =>

        // If looking at first observation
        if (trainingData.isEmpty) {

          // Add last subset
          model.addSubset(perceivedForm.get, subset)
        }

        // Add perceived form, spoken form, and frequency to trainingData
        trainingData += Tuple3(cells(1), cells(2), cells(3).toDouble)
    }
  }

  // Repeat EM algorithm for numTrial amount of trials
  for (trial <- 1 to numTrials) {

    // Print trial number
    System.out.println("Trial: " + trial)

    // Initialize log-likelihood
    var currLogLikelihood = 0.0
    var prevLogLikelihood = 0.0

    trial match {

      // Initialize weights to zero
      case 0 =>
        model.setParameters(0.0)

      //Initialize weights to one
      case 1 =>
        model.setParameters(1.0)

      case _ =>
    }

    // Reset parameters until we obtain a random initialization with log-likelihood better than randomThreshold
    if (allowRandomThreshold && trial > 1) {
      do {

        // Reset currLogLikelihood
        currLogLikelihood = 0.0

        // Randomize parameters for next trial run
        model.randomizeParameters

        // Find log-likelihood of the random initialization
        currLogLikelihood = this.findLogLikelihood
      }
      while (currLogLikelihood < randomThreshold)
    }
    else if (trial > 1) {

      // Randomize parameters for next trial run
      model.randomizeParameters
    }

    // Stop gradient optimization crashes from exiting the program
    try {
      // Use E-M Algorithm to calculate constraint weights that maximize the log likelihood
      do {

        // E-step calculates state occupancies using the forward-backward algorithm with the current transition functions.
        // Set prevLogLikelihood to last iteration's log-likelihood
        prevLogLikelihood = currLogLikelihood

        // F_a = Empirical count of features
        var F_a = new DenseTensor1(features.size, 0.0)

        // Iterate through training data
        for (instance <- trainingData) {

          // Store relevant factors and variables
          val relevantFactors = model.getSubsetFactors(instance._1)
          val relevantStates = model.getSubset(instance._1)

          // Forward probability
          val alpha = scala.collection.mutable.HashMap[String, Array[Double]]()

          // Backward probability
          val beta = scala.collection.mutable.HashMap[String, Array[Double]]()

          // Expected number of transitions from state i to state j, or factor(i,j)
          val eta = scala.collection.mutable.HashMap[MarkovTransition, Double]()

          // The probability of each state in terms of the complete path of the observation
          val probabilityOfParent = scala.collection.mutable.HashMap[State, Double]()

          // Initialize alpha and beta tables
          for (state <- model.getSubset(instance._1)) {
            alpha.put(state, Array.fill(T_states) {
              0.0
            })
            beta.put(state, Array.fill(T_states) {
              0.0
            })
          }

          // Sequentially find alpha(t) for all states
          for (t <- 0 to T_transitions) {
            t match {

              // Set initial state probability to 1
              case 0 => alpha(instance._1)(t) = 1.0

              // Set final state probability to 1
              case T_transitions => alpha(instance._2)(t) = 1.0

              // Find alpha(t) for t's in between
              case _ =>

                // Obtain normalization term
                var normalizationTerm = 0.0

                // Obtain alpha(t) for each state
                for (state <- relevantStates) {

                  alpha(state)(t) += model.parentFactors(States(state)).foldLeft(alpha(state)(t)) { (score, factor) =>
                    score + factor.score(factor.getChild.value, factor.getParent.value) *
                      alpha(factor.getParent.Value)(t - 1)
                  }

                  // Add alpha(t) to normalizationTerm
                  normalizationTerm += alpha(state)(t)
                }

                // Divide alpha(t) by normalizationTerm
                for (state <- relevantStates) {
                  alpha(state)(t) /= normalizationTerm
                }
            }
          }

          // Sequentially find beta(t) for all states
          for (t <- T_transitions to 0 by -1) {
            t match {

              // Set initial state probability to 1
              case 0 => beta(instance._1)(t) = 1.0

              // Set final state probability to 1
              case T_transitions => beta(instance._2)(t) = 1.0

              // Find beta(t) for t's in between
              case _ =>

                // Store normalization term
                var normalizationTerm = 0.0

                // Obtain beta(t) for each state
                for (state <- relevantStates) {
                  beta(state)(t) += model.childFactors(States(state)).foldLeft(beta(state)(t)) { (score, factor) =>
                    score + factor.score(factor.getChild.value, factor.getParent.value) *
                      beta(factor.getChild.Value)(t + 1)
                  }

                  // Add beta(t) to normalizationTerm
                  normalizationTerm += beta(state)(t)
                }

                // Divide beta(t) by normalizationTerm
                for (state <- relevantStates) {
                  beta(state)(t) /= normalizationTerm
                }
            }
          }

          // Obtain eta for all factors
          for (factor <- relevantFactors) {

            // Initialize eta value for a factor
            eta.put(factor, 0.0)

            // Obtain eta for a specific factor by summing across all t
            for (t <- 0 until T_transitions) {
              eta(factor) += alpha(factor.getParent.Value)(t) * factor.score(factor.getChild.value, factor.getParent.value) * beta(factor.getChild.Value)(t + 1)
            }
          }

          // Obtain the probability of each state in terms of the complete path of the observation
          var parents = scala.collection.mutable.Set[String]()
          var children = scala.collection.mutable.Set[String]()
          for (t <- 0 to T_states) {

            // Reset next variable
            children = children.empty

            // Obtain next viable states in path to find probability of
            for (parent <- parents) {
              for (factor <- model.childFactors(States(parent))) {
                children.add(factor.getChild.Value)
              }
            }

            // Reset prev variable
            parents = parents.empty
            t match {

              // Add probability of root state in the path
              case 0 => for (state <- relevantStates.filter(state => state == instance._1)) {
                probabilityOfParent(States(state)) = 1.0
                parents.add(state)
              }

              // Add probability of the states t transitions away from the root in the path
              case _ => for (state <- relevantStates.intersect(children)) {
                probabilityOfParent(States(state)) = model.parentFactors(States(state)).foldLeft(0.0) { (score, factor) => score + eta(factor) / model.childFactors(factor.getParent).foldLeft(0.0) { (normalizedEta, childFactor) => normalizedEta + eta(childFactor) } * probabilityOfParent(factor.getParent) }
                parents.add(state)
                }
            }
          }

          // Obtain this instance's contribution to F_a
          for (factor <- relevantFactors) {
            F_a += factor.weights * probabilityOfParent(factor.getParent) * eta(factor) / model.childFactors(factor.getParent).foldLeft(0.0) { (normalizedEta, childFactor) => normalizedEta + eta(childFactor) } * instance._3
          }
        }

        // M-step uses the GIS procedure with feature frequencies based on the E-step state occupancies to compute new transition functions
        val GIS = new ConjugateGradient(stepSize)

        // Create Weights Map from gradient for use with Factorie's ConjugateGradient
        val weightMap = new WeightsMap((Weights) => new DenseTensor1(features.size, 0.0))

        // Continue iterating until convergence
        while (!GIS.isConverged) {

          // E_a = Expected count of features
          var gradient_E_a = new DenseTensor1(features.size, 0.0)

          // Obtain this iteration's E_a (value and gradient) by looping through training data
          for (instance <- trainingData) {

            // Store relevant variables
            val relevantStates = model.getSubset(instance._1)

            // The probability of each state in terms of a complete path
            val probabilityOfParent = scala.collection.mutable.HashMap[State, Double]()

            // Obtain the probability of each state in terms of the complete path of the observation
            var parents = scala.collection.mutable.Set[String]()
            var children = scala.collection.mutable.Set[String]()
            for (t <- 0 to T_states) {

              // Reset next variable
              children = children.empty

              // Obtain next viable states in path to find probability of
              for (parent <- parents) {
                for (factor <- model.childFactors(States(parent))) {
                  children.add(factor.getChild.Value)
                }
              }

              // Reset prev variable
              parents = parents.empty
              t match {

                // Add probability of root state in the path
                case 0 => for (state <- relevantStates.filter(state => state == instance._1)) {
                  probabilityOfParent(States(state)) = 1.0
                  parents.add(state)
                }

                // Add probability of the states t transitions away from the root in the path
                case _ => for (state <- relevantStates.intersect(children)) {
                  probabilityOfParent(States(state)) = model.parentFactors(States(state)).foldLeft(0.0) { (score, factor) => score + factor.score(factor.getChild.value, factor.getParent.value) * probabilityOfParent(factor.getParent) }
                  parents.add(state)
                }
              }
            }

            for (factor <- model.getSubsetFactors(instance._1)) {
              gradient_E_a += factor.weights * probabilityOfParent(factor.getParent) * factor.score(factor.getChild.value, factor.getParent.value) * instance._3
            }
          }

          // Reset currLogLikelihood
          currLogLikelihood = 0.0

          // Find log-likelihood after one iteration of the EM algorithm
          currLogLikelihood = this.findLogLikelihood

          // Subtract regularization penalty to loss function
          currLogLikelihood -= (model.regularizationWeight / 2 * (model.constraintWeights.value dot model.constraintWeights.value)) - (model.negativePenalization / 2 * model.constraintWeights.value.foldLeft(0.0) { (weight, dim) => weight + pow(min(dim, 0), 2) })
          // Gradient of loss function
          val gradient = gradient_E_a - F_a - model.constraintWeights.value * model.regularizationWeight

          // Ensure gradient is positive for negative weights
          for (dim <- features.indices) {
            if (model.constraintWeights.value(dim) < 0)
              gradient(dim) = -model.constraintWeights.value(dim) * model.negativePenalization
          }

          // Update Weights Map with this iteration's gradient
          weightMap.update(model.constraintWeights, gradient)

          // Iterate
          GIS.step(model.parameters, weightMap, currLogLikelihood)
        }
      }

      // Repeat previous do block until tolerance is not surpassed
      while (abs(currLogLikelihood - prevLogLikelihood) > tolerance)

      // Remove all negatives (should all be barely negative)
      model.removeNegatives

      // Reset currLogLikelihood
      currLogLikelihood = 0.0

      // Find log-likelihood of the set of parameters without regularization penalties
      currLogLikelihood = this.findLogLikelihood

      // Append current trial
      trials += Tuple2(model.constraintWeights.value.asInstanceOf[DenseTensor1].copy, currLogLikelihood)
    }
    catch {
      case _: Throwable => System.err.println("Some ill parameter caused the gradient optimization to crash for trial: " + trial + ". This trial will be ignored.")
    }
  }

  // Ensure there are trials to print
  if (trials.isEmpty) {
    System.err.println("No results will be outputted because no trials have been successfully run.")
    System.exit(1)
  }

  // Obtain best trial
  val bestTrial = trials.maxBy(dim => dim._2)

  // Create file and writer to print out results
  val pw = new PrintWriter(new File("MaximumEntropyMarkovModel.txt"))

  // Print log-likelihood of best trial
  pw.write("Best Log-Likelihood\n")
  pw.write(bestTrial._2.toString  + "\n\n")

  // Print constraint weights of best trial
  pw.write("Constraint weights after optimization:\n")
  for ((feature, index) <- features.zipWithIndex) {
    pw.write(feature + "," + bestTrial._1(index) + "\n")
  }

  // Reset constraint parameters to best trial
  model.setParameters(bestTrial._1)

  // Print training and predicted probabilities of best trial
  pw.write("\nInput, Candidate, Observed Frequency, Observed Probability, Predicted Probability \n")
  for (instance <- trainingData.groupBy(instance =>instance._1).keys.toList) {

    // Store relevant factors and variables
    val relevantStates = model.getSubset(instance)

    // Forward probability
    val alpha = scala.collection.mutable.HashMap[String, Array[Double]]()

    // Initialize alpha and beta tables
    for (state <- model.getSubset(instance)) {
      alpha.put(state, Array.fill(T_states) {0.0})
    }

    // Sequentially find alpha(t) for all states
    for (t <- 0 to T_transitions) {
      t match {

        // Set initial state probability to 1
        case 0 => alpha(instance)(t) = 1.0

        // Find alpha(t)
        case _ =>

          // Obtain normalization term
          var normalizationTerm = 0.0

          // Obtain alpha(t) for each state
          for (state <- relevantStates) {
            alpha(state)(t) += model.parentFactors(States(state)).foldLeft(alpha(state)(t)) { (score, factor) =>
              score + factor.score(factor.getChild.value, factor.getParent.value) *
                alpha(factor.getParent.Value)(t - 1)
            }

            // Add alpha(t) to normalizationTerm
            normalizationTerm += alpha(state)(t)
          }

          // Divide alpha(t) by normalizationTerm
          for (state <- relevantStates) {
            alpha(state)(t) /= normalizationTerm

            // Obtain ground truth frequency and probability, and predicted probability
            if (model.childFactors(States(state)).isEmpty && t == T_transitions) {
              val trainingFrequency = trainingData.find(element => element._1 == instance && element._2 == state) match {

                // Parse for ground truth frequency
                case Some(element) => element._3
                case None => 0.0
              }

              // Normalize ground truth frequencies into probabilities
              val normalizationTerm = trainingData.filter(item => item._1 == instance).foldLeft(0.0) {(normalizationTerm, item) => normalizationTerm + item._3}

              // Since paths may be different sizes, we will treat the last non-zero alpha(t) as the end of the path
              var finalAlpha = 0.0
              for (t <- 0 to T_transitions) {
                if (alpha(state)(t) > 0.0) {
                  finalAlpha = alpha(state)(t)
                }
              }

              // If we want to remove markers
              if (removeMarkers) {

                """^\([0-9]+\)""".r.findFirstIn(state) match {

                  // If the string has a marker
                  case Some(marker) =>

                    // Print training and predicted probabilities of best trial
                    pw.write(instance + "," + state.drop(marker.length) + "," + trainingFrequency + "," + trainingFrequency / normalizationTerm + "," + finalAlpha + "\n")

                  // If the string does not have a marker
                  case None =>

                    // Print training and predicted probabilities of best trial
                    pw.write(instance + "," + state + "," + trainingFrequency + "," + trainingFrequency / normalizationTerm + "," + finalAlpha + "\n")
                }
              }
              else {

                // Print training and predicted probabilities of best trial
                pw.write(instance + "," + state + "," + trainingFrequency + "," + trainingFrequency / normalizationTerm + "," + finalAlpha + "\n")

              }
            }
          }
      }
    }
  }

  // Close file writer
  pw.close()

  // Find the logLikelihood
  def findLogLikelihood = {

    // Initialize currLogLikelihood
    var currLogLikelihood = 0.0

    // Find this iteration's log-likelihood
    for (instance <- trainingData) {

      // Store relevant factors and variables
      val relevantStates = model.getSubset(instance._1)

      // Forward probability
      val alpha = scala.collection.mutable.HashMap[String, Array[Double]]()

      // Initialize alpha and beta tables
      for (state <- model.getSubset(instance._1)) {
        alpha.put(state, Array.fill(T_states) {0.0})
      }
      // Sequentially find alpha(t) for all states
      for (t <- 0 to T_transitions) {
        t match {

          // Set initial state probability to 1
          case 0 => alpha(instance._1)(t) = 1.0

          // Find alpha(t)
          case _ =>

            // Obtain normalization term
            var normalizationTerm = 0.0

            // Obtain alpha(t) for each state
            for (state <- relevantStates) {

              alpha(state)(t) += model.parentFactors(States(state)).foldLeft(alpha(state)(t)) { (score, factor) =>
                score + factor.score(factor.getChild.value, factor.getParent.value) *
                  alpha(factor.getParent.Value)(t - 1)
              }

              // Add alpha(t) to normalizationTerm
              normalizationTerm += alpha(state)(t)
            }

            // Divide alpha(t) by normalizationTerm
            for (state <- relevantStates) {
              alpha(state)(t) /= normalizationTerm
            }
        }
      }

      // Add log of real frequency multiplied by probability of seeing this form
      currLogLikelihood += instance._3 * log(alpha(instance._2)(T_transitions))
    }

    // Return currLogLikelihood
    currLogLikelihood
  }
}
