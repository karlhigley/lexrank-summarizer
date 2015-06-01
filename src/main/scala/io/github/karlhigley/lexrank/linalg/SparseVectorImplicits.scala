package io.github.karlhigley.lexrank.linalg

import org.apache.spark.mllib.linalg.SparseVector

object SparseVectorImplicits {
   implicit def SparseVector2SparseVectorOps(vector: SparseVector) = new SparseVectorOps(vector.size, vector.indices, vector.values) 
}
