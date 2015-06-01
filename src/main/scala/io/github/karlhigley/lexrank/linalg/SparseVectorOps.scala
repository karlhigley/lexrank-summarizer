package io.github.karlhigley.lexrank.linalg

import org.apache.spark.mllib.linalg.SparseVector

class SparseVectorOps(val size: Int, val indices: Array[Int], val values: Array[Double]) {
  def mapValues(f: Double => Double): SparseVector = {
    new SparseVector(size, indices, values.map(f))
  }
}