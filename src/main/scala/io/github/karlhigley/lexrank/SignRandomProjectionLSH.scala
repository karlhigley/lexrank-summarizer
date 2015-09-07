package io.github.karlhigley.lexrank

import scala.collection.immutable.BitSet
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.util.hashing.MurmurHash3

import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.Logging

class SignRandomProjectionLSH(poolSize: Int = 10000) extends Serializable with Logging {
  val pool = SignRandomProjectionLSH.generatePool(poolSize)
  
  def computeSignature(vector: SparseVector, length: Int): BitSet = {
    val buf = ArrayBuffer.empty[Int]
    
    val elements = vector.indices.zip(vector.values)
    for (bit <- 1 to length) {
      val components = elements.map(e => {
          val hash      = MurmurHash3.productHash((bit, e._1))
          val poolIndex = ((hash % poolSize) + poolSize) % poolSize
          val result    = e._2 * pool(poolIndex)
          result
      })

      val dotProduct = components.reduce(_ + _)
      if (dotProduct > 0) {
        buf += bit
      }
    }

    BitSet(buf.toArray:_*)
  }
}

object SignRandomProjectionLSH {
  def signatureSet(length: Int): Set[BitSet] = {
    BitSet(1 to length:_*).subsets.toSet
  }

  private def generatePool(size: Int): Array[Double] = {
    val rand = new Random()
    val buf  = ArrayBuffer.fill[Double](size)(rand.nextGaussian)
    buf.toArray
  }
}