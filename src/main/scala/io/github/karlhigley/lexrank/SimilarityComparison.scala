package io.github.karlhigley.lexrank

import scala.math.{log, max}

import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.linalg.{Vector, SparseVector}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}

class SimilarityComparison(threshold: Double, buckets: Int) extends Serializable {
  val signatureGen = new CosineLSH

  def apply(sentences: RDD[SentenceFeatures]): RDD[SentenceComparison] = {
    val maxColumn = sentences.map(_.id).reduce(math.max(_, _)) + 1

    val matrices = bucketSentences(sentences, buckets).flatMap(buildRowMatrix(_, 1 << 20, maxColumn))
    matrices.foreach(_.rows.persist())

    val similarities = matrices
                          .map(computeSimilarities(_, threshold))
                          .reduce(_ union _)
                          .coalesce(sentences.partitions.size)

    similarities.persist()
    similarities.count()

    matrices.foreach(_.rows.unpersist())

    similarities.flatMap(MatrixEntry.unapply(_)).map(SentenceComparison.tupled)
  }

  private def computeSimilarities(matrix: RowMatrix, threshold: Double): RDD[MatrixEntry] = {
    matrix.columnSimilarities(threshold).entries.filter(_.value > threshold)
  }

  private def bucketSentences(sentences: RDD[SentenceFeatures], buckets: Int) = {
    val signatureBits = (log(buckets)/log(2)).toInt
    
    val signatures    = sentences.map(s => {
      (signatureGen.computeSignature(s.features, signatureBits), s)
    }).cache()
    signatures.count()
    
    CosineLSH
      .signatureSet(signatureBits)
      .map(k => {
        signatures.filter(s => s._1 == k).map(s => s._2)
      })
  }

  private def buildRowMatrix(columns: RDD[SentenceFeatures], rowCount: Long, colCount: Long) : Option[RowMatrix] = {
    val matrixEntries = columns.flatMap {
      case SentenceFeatures(colNum, _, vector) =>
        sparseElements(vector).map {
          case (rowNum, value) => MatrixEntry(rowNum, colNum, value)
        }
    }

    matrixEntries.isEmpty() match {
      case true => None
      case _    => Some(new CoordinateMatrix(matrixEntries, rowCount, colCount).toRowMatrix())
    }    
  }

  private def sparseElements(vector: SparseVector): Seq[(Int, Double)] = {
    vector.indices.zip(vector.values)
  }
}