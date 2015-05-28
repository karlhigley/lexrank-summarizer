package io.github.karlhigley.lexrank.linalg

import java.util.Random

import scala.collection.mutable.ListBuffer

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Vector, DenseVector, SparseVector}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD

class RowMatrixOps(val rows: RDD[Vector]) {
  def columnSimilarities(colMags: SparseVector, threshold: Double): CoordinateMatrix = {
    require(threshold >= 0, s"Threshold cannot be negative: $threshold")

    /*
    if (threshold > 1) {
      
      logWarning(s"Threshold is greater than 1: $threshold " +
      "Computation will be more efficient with promoted sparsity, " +
      " however there is no correctness guarantee.")
      
    }
    */

    val gamma = if (threshold < 1e-6) {
      Double.PositiveInfinity
    } else {
      10 * math.log(colMags.indices.length) / threshold
    }

    columnSimilaritiesDIMSUM(colMags, gamma)
  }

  private def columnSimilaritiesDIMSUM(colMags: SparseVector, gamma: Double): CoordinateMatrix = {
    require(gamma > 1.0, s"Oversampling should be greater than 1: $gamma")
    //require(colMags.size == this.numCols(), "Number of magnitudes didn't match column dimension")
    val sg = math.sqrt(gamma) // sqrt(gamma) used many times

    // Don't divide by zero for those columns with zero magnitude
    val colMagsCorrected = new SparseVector(colMags.size, colMags.indices, colMags.values.map(x => if (x == 0) 1.0 else x))

    val sc = rows.context
    val pBV = sc.broadcast(new SparseVector(colMags.size, colMags.indices, colMags.values.map(c => sg / c)))
    val qBV = sc.broadcast(new SparseVector(colMags.size, colMags.indices, colMags.values.map(c => math.min(sg, c))))

    val sims = rows.mapPartitionsWithIndex { (indx, iter) =>
      val p = pBV.value
      val q = qBV.value

      val rand = new Random(indx)
      val scaled = new Array[Double](p.size)
      iter.flatMap { row =>
        val buf = new ListBuffer[((Int, Int), Double)]()
        row match {
          case SparseVector(size, indices, values) =>
            val nnz = indices.size
            var k = 0
            while (k < nnz) {
              scaled(k) = values(k) / q(indices(k))
              k += 1
            }
            k = 0
            while (k < nnz) {
              val i = indices(k)
              val iVal = scaled(k)
              if (iVal != 0 && rand.nextDouble() < p(i)) {
                var l = k + 1
                while (l < nnz) {
                  val j = indices(l)
                  val jVal = scaled(l)
                  if (jVal != 0 && rand.nextDouble() < p(j)) {
                    buf += (((i, j), iVal * jVal))
                  }
                  l += 1
                }
              }
              k += 1
            }
          /*
          case DenseVector(values) =>
            val n = values.size
            var i = 0
            while (i < n) {
              scaled(i) = values(i) / q(i)
              i += 1
            }
            i = 0
            while (i < n) {
              val iVal = scaled(i)
              if (iVal != 0 && rand.nextDouble() < p(i)) {
                var j = i + 1
                while (j < n) {
                  val jVal = scaled(j)
                  if (jVal != 0 && rand.nextDouble() < p(j)) {
                    buf += (((i, j), iVal * jVal))
                  }
                  j += 1
                }
              }
              i += 1
            }
            */
        }
        buf
      }
    }.reduceByKey(_ + _).map { case ((i, j), sim) =>
      MatrixEntry(i.toLong, j.toLong, sim)
    }
    new CoordinateMatrix(sims, colMags.size, colMags.size)
  }
}