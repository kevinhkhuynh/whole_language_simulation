package com.zuraw.scala

import cc.factorie.variable.{CategoricalDomain, CategoricalVariable}

/**
  * Created by kevinhuynh on 2/3/18.
  */
// A domain and variable type for storing states (the forms of a chosen word)
object StateDomain extends CategoricalDomain[String]
class State(str:String) extends CategoricalVariable(str) {
  def domain = StateDomain
  def Value = str
}
