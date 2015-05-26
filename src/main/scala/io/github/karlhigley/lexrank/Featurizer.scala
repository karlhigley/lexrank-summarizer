package io.github.karlhigley.lexrank

import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.SparseVector

case class SentenceFeatures(id: Long, docId: String, features: SparseVector)

class Featurizer extends Serializable {
  def apply(tokens: RDD[SentenceTokens]) : RDD[SentenceFeatures] = {
    val hashingTF  = new HashingTF()
    val idfModel   = new IDF(minDocFreq = 2)

    val termFrequencies = tokens.map(t => {
        (t.id, t.docId, hashingTF.transform(t.tokens))
    })
    
    val idf = idfModel.fit(termFrequencies.map({ case (_, _, tf) => tf }))

    termFrequencies
      .map({
        case (id, docId, tf) =>
          val tfidfVector = idf.transform(tf).asInstanceOf[SparseVector]
          val (indices, values) = tfidfVector.indices.zip(tfidfVector.values).filter(_._2 > 0.0).unzip
          val featureVector = new SparseVector(tfidfVector.size, indices.toArray, values.toArray)
          SentenceFeatures(id, docId, featureVector)
      })
      .filter(_.features.indices.size > 0)
  }
}