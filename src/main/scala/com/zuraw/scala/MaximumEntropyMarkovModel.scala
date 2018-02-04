package com.zuraw.scala

import cc.factorie.la.DenseTensor1
import cc.factorie.model.{ItemizedModel, Parameters}
import com.zuraw.scala.WholeLanguageSimulation.{States, features, random}

import scala.math.abs

/**
  * Created by kevinhuynh on 2/3/18.
  */
// Maximum Entropy Markov Model
class MaximumEntropyMarkovModel extends ItemizedModel with Parameters {

  // Constraint weights (initialized to 0 plus some Gaussian noise)
  var constraintWeights = Weights(new DenseTensor1(Array.fill(features.size)(0.0).map{dim => dim + abs(random.nextGaussian())}))

  // Regularization weight (initialized to 10)
  val regularizationWeight = 10.0

  // Extremely penalize negative weights (Our situation requires positive weights)
  val negativePenalization = 10000.0

  // Holds the different subsets of variables that equate to different word chains
  var subset = scala.collection.mutable.HashMap[String, scala.collection.mutable.Set[String]]()

  // Randomize constraint weights
  def randomizeParameters = {
    this.constraintWeights = Weights(new DenseTensor1(Array.fill(features.size)(0.0).map{dim => dim + abs(random.nextGaussian())}))
  }

  def setParameters(parameters: DenseTensor1) = {
    this.constraintWeights = Weights(new DenseTensor1(parameters))
  }

  // Add a new word chain, or subset of variables, to the model
  def addSubset(perceivedForm: String, relatedForms: scala.collection.mutable.Set[String]) = subset.put(perceivedForm, relatedForms)

  // Retrieve a word chain, or subset of variables, from the model
  def getSubset(perceivedForm: String) = subset.getOrElse(perceivedForm, scala.collection.mutable.Set[String]())

  // Retrieve a word chain's factors, or all factors relating to a subset of variables, from the model
  def getSubsetFactors(perceivedForm: String) = this.factors(this.getSubset(perceivedForm).map{element => States.getOrElse(element, new State(element))}).asInstanceOf[Iterable[MarkovTransition]]

  // Return all transitions that are emitted by a certain state
  def childFactors(parent: State):Iterable[MarkovTransition] = this.factors(parent).asInstanceOf[Iterable[MarkovTransition]].filter(factor => factor.getParent == parent)

  // Return all transitions that are emitted by a certain state
  def parentFactors(child: State):Iterable[MarkovTransition] = this.factors(child).asInstanceOf[Iterable[MarkovTransition]].filter(factor => factor.getChild == child)
}