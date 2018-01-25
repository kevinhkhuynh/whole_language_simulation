package com.zuraw.scala

import scala.math._
import scala.collection.JavaConverters._
import cc.factorie._            // The base library: variables, factors
import cc.factorie.variable._
import cc.factorie.model._
import cc.factorie.infer._
import cc.factorie.la          // Linear algebra: tensors, dot-products, etc.
import cc.factorie.optimize._   // Gradient-based optimization and training

/**
  * Created by kevinhuynh on 1/18/18.
  */
object WholeLanguageSimulation extends App {

  // Read tableau file
  var tableau = readFile.read("FullGrammar_forOTSoft10.csv").asScala

  // A domain and variable type for storing features (the constraints)
  object FeatureDomain extends CategoricalDomain[String]
  class Feature(str:String) extends CategoricalVariable(str) { def domain = FeatureDomain }
  val features = scala.collection.mutable.Set[Feature]()

  // Define constraints
  for (feature <- tableau.head) yield {
    features.add(new Feature(feature))
  }

  // A domain and variable type for storing states (the forms of a chosen word)
  object StateDomain extends CategoricalDomain[String]
  class State(str:String) extends CategoricalVariable(str) {
    def domain = StateDomain
    def Value = str.asInstanceOf[CategoricalValue[String]]
  }
  val States = scala.collection.mutable.HashMap[String, State]()

  // Define model
  class MaximumEntropyMarkovModel extends ItemizedModel with Parameters {

    // Lambdas, or constraint weights
    def constraintWeights = Weights(new DenseTensor1(features.size, 1.0))

    // Return all transitions that are emitted by a certain state
    def childFactors(p1: State):Iterable[markovTransition] =
      this.factors(p1).asInstanceOf[Iterable[markovTransition]].filter(factor => factor.getParent == p1)
  }
  val model = new MaximumEntropyMarkovModel

  // A type for storing transitions
  class markovTransition(child: State, parent: State, featureTensor:DenseTensor1) extends Factor2(child: State, parent: State) {

    // The feature values for a given X -> Y
    def weights = featureTensor

    def getParent = parent
    def getChild = child

    // Return P*(x), or maxEnt score
    def maxEntScore = exp(-(model.constraintWeights.value dot weights))

    // Return P(x), or normalized probability. v1 and p1 are unused, but mandated by Factorie
    def score(v1: child.Value, p1: parent.Value) = {

      // Obtain all child transitions from the parent of this transition
      val childFactors = model.childFactors(parent)

      // Find normalization constant
      var z = 0.0
      for (factor <- childFactors) {
        z += factor.maxEntScore
      }

      // Obtain normalized probability
      this.maxEntScore / z
    }
  }

  // Define states, transitions, and candidate weights, and then add them to the model
  for (line <- tableau.drop(1)) yield {
    val cells = line.mkString(",").split(",")

    // Have to add a symbol to differentiate between surface forms that are perceived or created
    if (cells(0)(0) == '|')
      cells(1) = "R" + cells(1)

    // Parse through tableau to obtain X, Y (X->Y) and feature values
    val candidateWeights = new DenseTensor1(features.size, 0.0)
    for ((element, index) <- cells.zipWithIndex) {
      index match {
        case 0 => if (States.get(cells(0)).isEmpty)
            States.put(cells(0), new State(cells(0)))
        case 1 => if (States.get(cells(1)).isEmpty)
          States.put(cells(1), new State(cells(1)))
        // TODO: We can use case 2 to obtain frequencies if needed
        case 2 =>
        case _ =>
          element match {
            case "" => candidateWeights(index - 3) = 0
            case nonBlank => candidateWeights(index - 3) = nonBlank.toDouble
          }
      }
    }

    //Add appropriate transition to the model
    model += new markovTransition(States.getOrElse(cells(1), new State(cells(1))), States.getOrElse(cells(0), new State(cells(0))), candidateWeights)
  }




  //println(model.childFactors(States.getOrElse("/(mno)ka(lme)/", new State("asas"))))


  //  class transition(child: State, parent: State, featureTensor:DenseTensor1) extends DirectedFactorWithStatistics2(child: State, parent: State) {
  //    val candidateWeights = featureTensor
  //    val publicChild = child
  //    def unNormalizedPr:Double = exp(-(model.constraintWeights dot candidateWeights))
  //    def pr(v1: child.Value, p1: parent.Value) = {
  //      //val childFactors = model.childFactors(new State(p1.toString()))
  //      var z = 0.0
  //      //for (factor <- childFactors) {
  //      //  z += factor.asInstanceOf[transition].unNormalizedPr
  //      //}
  //      //this.unNormalizedPr / z
  //      z
  //    }
  //    def sampledValue(p1: parent.Value)(implicit random: Random) = publicChild.value
  //  }

  /*

  }}
  val labelSequences = for (sentence <- data) yield new LabelSeq ++= sentence.split(" ").map(s => {
    val a = s.split("/")
    new Label(a(1), new Token(a(0)))
  })
  // Define a model structure
  val model = new Model with Parameters {
    // Two families of factors, where factor scores are dot-products of sufficient statistics and weights.
    // (The weights will set in training below.)
    val markov = new DotFamilyWithStatistics2[Label,Label] {
      val weights = Weights(new la.DenseTensor2(LabelDomain.size, LabelDomain.size))
    }
    val observ = new DotFamilyWithStatistics2[Label,Token] {
      val weights = Weights(new la.DenseTensor2(LabelDomain.size, TokenDomain.size))
    }
    // Given some variables, return the collection of factors that neighbor them.
    def factors(labels:Iterable[Var]) = labels match {
      case labels:LabelSeq =>
        labels.map(label => new observ.Factor(label, label.token))
        ++ labels.sliding(2).map(window => new markov.Factor(window.head, window.last))
    }
  }
  // Learn parameters
  val trainer = new BatchTrainer(model.parameters, new ConjugateGradient)
  trainer.trainFromExamples(labelSequences.map(labels => new LikelihoodExample(labels, model, InferByBPChain)))
  // Inference on the same data.  We could let FACTORIE choose the inference method,
  // but here instead we specify that is should use max-product belief propagation
  // specialized to a linear chain
  labelSequences.foreach(labels => BP.inferChainMax(labels, model))
  // Print the learned parameters on the Markov factors.
  println(model.markov.weights)
  // Print the inferred tags
  labelSequences.foreach(_.foreach(l => println(s"Token: ${l.token.value} Label: ${l.value}")))*/
  }