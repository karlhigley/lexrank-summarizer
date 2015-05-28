package io.github.karlhigley.lexrank.linalg

import org.apache.spark.mllib.linalg.distributed.RowMatrix

object RowMatrixImplicits {
   implicit def RowMatrix2RowMatrixOps(matrix : RowMatrix) = new RowMatrixOps(matrix.rows) 
}
