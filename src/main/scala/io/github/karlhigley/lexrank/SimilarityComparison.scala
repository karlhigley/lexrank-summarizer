package io.github.karlhigley.lexrank

import scala.math.{log, max, sqrt}

import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.linalg.{Vectors, SparseVector}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}

import io.github.karlhigley.lexrank.linalg.RowMatrixImplicits._

class SimilarityComparison(threshold: Double, buckets: Int) extends Serializable {
  val signatureGen = new CosineLSH

  def apply(sentences: RDD[SentenceFeatures]): RDD[SentenceComparison] = {
    val numCols = sentences.map(_.id).reduce(math.max(_, _)) + 1

    val matricesWithMags = bucketSentences(sentences, buckets).filter(!_.isEmpty()).map(buildRowMatrix(_, 1 << 20, numCols))
    matricesWithMags.foreach(_._1.rows.persist())

    val similarities = matricesWithMags
                          .map({
                            case (matrix, mags) => computeSimilarities(matrix, mags, threshold)
                          })
                          .reduce(_ union _)
                          .repartition(sentences.partitions.size)

    similarities.persist()
    similarities.count()

    matricesWithMags.foreach(_._1.rows.unpersist())

    similarities.flatMap(MatrixEntry.unapply(_)).map(SentenceComparison.tupled)
  }

  private def computeSimilarities(matrix: RowMatrix, colMags: SparseVector, threshold: Double): RDD[MatrixEntry] = {
    matrix.columnSimilarities(colMags, threshold).entries.filter(_.value > threshold)
  }

  private def bucketSentences(sentences: RDD[SentenceFeatures], buckets: Int): Set[RDD[SentenceFeatures]] = {
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

  private def buildRowMatrix(columns: RDD[SentenceFeatures], rowCount: Long, colCount: Long) : (RowMatrix, SparseVector) = {
    val colMags   = computeMagnitudes(columns, colCount.toInt)

    val matrixEntries = columns.flatMap {
      case SentenceFeatures(colNum, _, vector) =>
        sparseElements(vector).map {
          case (rowNum, value) => MatrixEntry(rowNum, colNum, value)
        }
    }

    (new CoordinateMatrix(matrixEntries, rowCount, colCount).toRowMatrix(), colMags)
  }

  private def sparseElements(vector: SparseVector): Seq[(Int, Double)] = {
    vector.indices.zip(vector.values)
  }

  private def computeMagnitudes(sentences: RDD[SentenceFeatures], numCols: Int): SparseVector = {
    val pairs = sentences.map(f => (f.id.toInt, Vectors.norm(f.features, 2))).sortByKey().collect()
    val (indices, values) = pairs.toList.unzip
    new SparseVector(numCols, indices.toArray, values.toArray)
  }
}