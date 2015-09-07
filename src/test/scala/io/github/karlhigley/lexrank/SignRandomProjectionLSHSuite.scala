package io.github.karlhigley.lexrank

import org.scalatest.FunSuite

import org.apache.spark.mllib.linalg.SparseVector

class SignRandomProjectionLSHSuite extends FunSuite with TestSparkContext {
  val lshModel = new SignRandomProjectionLSH

  test("signatures are deterministic") {
    val nonzero = new SparseVector(3, Array(1,2,3), Array(1,1,1))
    val signatureA = lshModel.computeSignature(nonzero, 2)
    val signatureB = lshModel.computeSignature(nonzero, 2)
    assert(signatureA === signatureB)
  }

  test("zero vectors get signatures") {
    val zero = new SparseVector(3, Array(1,2,3), Array(0,0,0))
    val signature0 = lshModel.computeSignature(zero, 2)
  }
}