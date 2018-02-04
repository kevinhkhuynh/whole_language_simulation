package com.zuraw.scala

import cc.factorie.la.DenseTensor1
import cc.factorie.model.Factor2
import com.zuraw.scala.WholeLanguageSimulation.model

import scala.math.exp

/**
  * Created by kevinhuynh on 2/3/18.
  */
// A type for storing transitions
class MarkovTransition(child: State, parent: State, featureTensor:DenseTensor1) extends Factor2(child: State, parent: State) {

  // The feature values for a given X -> Y
  def weights = featureTensor

  // Obtain parent or child states
  def getParent = parent
  def getChild = child

  // Return P*(x), or maxEnt score
  def maxEntScore = exp(-(model.constraintWeights.value dot weights))

  // Return P(x), or normalized probability. v1 and p1 are unused, but mandated by Factorie
  def score(v1: child.Value, p1: parent.Value) = this.maxEntScore / model.childFactors(parent).foldLeft(0.0){(z, factor) => z + factor.maxEntScore}
}