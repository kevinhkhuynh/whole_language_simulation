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

  // Create a random variable for Gaussian noise
  implicit val random = new scala.util.Random()

  // The length of the word chain path in number of transitions and states
  val T_transitions = 3
  var T_states = 4

  // Read tableau file
  var tableau = readFile.read("FullGrammar_forOTSoft_plus_missing_intermediate_states.csv").asScala

  // Variable to store features (the constraints)
  val features = scala.collection.mutable.ArrayBuffer[String]()

  // Define constraints
  tableau.head.foreach{feature => features.append(feature)}

  // Define states
  val States = scala.collection.mutable.HashMap[String, State]()

  // Define model
  val model = new MaximumEntropyMarkovModel

  // Variable holding perceived form so that states for different word chains can be kept separate
  var perceivedForm = None:Option[String]

  // Hold current subset of variables, or word chain
  var subset = scala.collection.mutable.Set[String]()

  // Hold training data that consists of [perceivedForm, spokenForm, count of spokenForms in data]
  var trainingData = scala.collection.mutable.ArrayBuffer[Tuple3[String, String, Double]]()

  // Define states, transitions, and candidate weights, and then add them to the model
  for (line <- tableau.drop(1)) {

    // Split line into cells
    val cells = line.mkString(",").split(",")

    cells(0)(0) match {

      // Have to add a symbol to differentiate between surface forms that are perceived or created
      case '|' => cells(1) = "R" + cells(1)

      // Change perceived form when starting new word chain
      case '[' =>

        // Save subset and empty for a new word chain
        if (perceivedForm.isDefined && perceivedForm.get != cells(0)) {
          model.addSubset(perceivedForm.get, subset)
          subset = subset.empty
        }
        perceivedForm = Some(cells(0))

      case _ =>
    }

    // Parse through tableau to obtain X, Y (X->Y) and feature values
    val candidateWeights = new DenseTensor1(features.size, 0.0)
    for ((element, index) <- cells.zipWithIndex) {
      index match {

        // Add X and Y to list of variables and to current word chain
        case 0 => if (States.get(cells(index)).isEmpty) {
          States.put(cells(index), new State(cells(index)))
          subset.add(cells(index))
        }
        case 1 => if (States.get(cells(index)).isEmpty) {
          States.put(cells(index), new State(cells(index)))
          subset.add(cells(index))
        }

        // Add word chain to training data
        case 2 => if (cells(2).toDouble > 0) {
          trainingData += Tuple3(perceivedForm.get, cells(1), cells(index).toDouble)
        }

        // Parse the features of the tableau
        case _ => element match {
          case "" => candidateWeights(index - 3) = 0
          case nonBlank => candidateWeights(index - 3) = nonBlank.toDouble
        }
      }
    }

    model += new MarkovTransition(States(cells(1)), States(cells(0)), candidateWeights)
  }

  // Add last subset
  model.addSubset(perceivedForm.get, subset)

  // Run numTrials amount of trials
  val numTrials = 1000

  // Store learned constraint weights and negative log-likelihood for a given trial
  var trials = scala.collection.mutable.ArrayBuffer[Tuple2[DenseTensor1, Double]]()

  // Choose tolerance to end EM algorithm
  val tolerance = 1.0E-4

  // Repeat EM algorithm for numTrial amount of trials
  for (trial <- 0 to numTrials) {

    // Randomize parameters for next trial run
    model.randomizeParameters

    // Initialize log-likelihood
    var currLogLikelihood = 0.0
    var prevLogLikelihood = 0.0

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

        // Initialize alpha and beta tables
        for (state <- model.getSubset(instance._1)) {
          alpha.put(state, Array.fill(T_states) {0.0})
          beta.put(state, Array.fill(T_states) {0.0})
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

        // Obtain this instance's contribution to F_a
        for (factor <- relevantFactors) {
          F_a += factor.weights * eta(factor) / model.childFactors(factor.getParent).foldLeft(0.0) { (normalizedEta, childFactor) => normalizedEta + eta(childFactor) } * instance._3
        }
      }

      // M-step uses the GIS procedure with feature frequencies based on the E-step state occupancies to compute new transition functions
      val GIS = new ConjugateGradient

      // Create Weights Map from gradient for use with Factorie's ConjugateGradient
      val weightMap = new WeightsMap((Weights) => new DenseTensor1(features.size, 0.0))

      // Continue iterating until convergence
      while (!GIS.isConverged) {

        // E_a = Expected count of features
        var gradient_E_a = new DenseTensor1(features.size, 0.0)
        var value_E_a = 0.0

        // Obtain this iteration's E_a (value and gradient) by looping through training data
        for (instance <- trainingData) {
          var temp = 0.0
          for (factor <- model.getSubsetFactors(instance._1)) {
            gradient_E_a += factor.weights * factor.score(factor.getChild.value, factor.getParent.value) * instance._3
            temp += factor.maxEntScore
          }
          value_E_a += log(temp) * instance._3
        }

        // Loss function
        val value = (F_a dot model.constraintWeights.value) - value_E_a - (model.regularizationWeight / 2 * (model.constraintWeights.value dot model.constraintWeights.value)) - (model.negativePenalization / 2 * model.constraintWeights.value.foldLeft(0.0) { (weight, dim) => weight + pow(min(dim, 0), 2) })

        // Gradient of loss function
        val gradient = F_a - gradient_E_a - (model.constraintWeights.value * model.regularizationWeight) - (new DenseTensor1(model.constraintWeights.value.map { dim => max(-dim, 0) }) * model.negativePenalization)

        // Update Weights Map with this iteration's gradient
        weightMap.update(model.constraintWeights, gradient)

        // Iterate
        GIS.step(model.parameters, weightMap, value)
      }

      // Reset currLogLikelihood
      currLogLikelihood = 0.0

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

        // Add negative log of real frequency multiplied by probability of seeing this form
        currLogLikelihood += instance._3 * log(alpha(instance._2)(T_transitions))
      }
    }

    // Repeat previous do block until tolerance is not surpassed
    while (currLogLikelihood - prevLogLikelihood > tolerance)

    // Append current trial
    trials += Tuple2(model.constraintWeights.value.asInstanceOf[DenseTensor1].copy, currLogLikelihood)
  }

  // Obtain best trial
  val bestTrial = trials.maxBy(dim => dim._2)
  
  // Create file and writer to print out results
  val pw = new PrintWriter(new File("MaximumEntropyMarkovModel.txt"))

  // Print negative log-likelihood of best trial
  pw.write("Best Negative Log-Likelihood \n")
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
            if (state(0) == 'R' && t == T_transitions) {
              val trainingFrequency = trainingData.find(element => element._1 == instance && element._2 == state) match {

                // Parse for ground truth frequency
                case Some(element) => element._3
                case None => 0.0
              }

              // Normalize ground truth frequencies into probabilities
              val normalizationTerm = trainingData.filter(item => item._1 == instance).foldLeft(0.0) {(normalizationTerm, item) => normalizationTerm + item._3}

              // Print training and predicted probabilities of best trial
              pw.write(instance + "," +state + "," + trainingFrequency + "," + trainingFrequency / normalizationTerm + "," +alpha(state)(t)  + "\n")
            }
          }
      }
    }
  }

  // Close file writer
  pw.close()
}