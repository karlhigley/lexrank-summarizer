package io.github.karlhigley.lexrank

import org.apache.spark.SparkContext

import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.{SparseVector, Vector}

case class SentenceFeatures(id: Long, docId: String, features: SparseVector)

class Featurizer(numStopwords: Int = 0) extends Serializable {
  private val hashingTF = new HashingTF()
  private val byIDF = Ordering[Double].on[(Int,Double)](_._2)

  def apply(tokens: RDD[SentenceTokens]) : RDD[SentenceFeatures] = {
    val idf = new IDF(minDocFreq = 2)

    val termFrequencies = tokens.map(t => {
        (t.id, t.docId, hashingTF.transform(t.tokens))
    })
    
    val idfModel = idf.fit(termFrequencies.map({ case (_, _, tf) => tf }))

    val stopwordIndices = identifyStopwords(idfModel.idf.toSparse, numStopwords)

    termFrequencies
      .map({
        case (id, docId, tf) =>
          val tfidf = idfModel.transform(tf).toSparse
          val features = removeStopwords(tfidf, stopwordIndices)
          SentenceFeatures(id, docId, features)
      })
      .filter(_.features.indices.size > 0)
  }

  def indexOf(token: String): Int = {
    hashingTF.indexOf(token)
  }

  private def identifyStopwords(idf: SparseVector, numStopwords: Int) = {
    featureTuples(idf).sorted(byIDF).take(numStopwords).map(_._1)
  }

  private def removeStopwords(tf: SparseVector, stopwordIndices: Array[Int]) = {
    val (indices, values) =
        featureTuples(tf)
          .filter(p => !stopwordIndices.contains(p._1))
          .unzip
    new SparseVector(tf.size, indices.toArray, values.toArray)
  }

  private def featureTuples(featureVector: SparseVector) = {
    featureVector.indices.zip(featureVector.values)
  }
}