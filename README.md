# LexRank Summarizer

This is a Spark-based extractive summarizer, based on the [LexRank algorithm](http://arxiv.org/pdf/1109.2128.pdf).  It extracts a 5 sentence summary from each document in the corpus.

Boilerplate language is detected across documents in the corpus using pairwise comparisons of sentence similarity.  The similarity computations use the [DIMSUM algorithm](http://arxiv.org/abs/1304.1467) to reduce the computational effort involved.

## Usage

Build a JAR file from the source with `sbt assembly`.  Submit a job to Spark with:

```
spark-submit --class io.github.karlhigley.lexrank.Driver <path to jar file> [options]

Options:
-i PATH,  --input PATH         Relative path of input files (default: "./input")
-o PATH,  --output PATH        Relative path of output files (default: "./output")
-s PATH,  --stopwords PATH     Relative path of stopwords file (default: "./stopwords")
-l VALUE, --length VALUE       Number of sentences to extract from each document (default: 5) 
-b VALUE, --boilerplate VALUE  Similarity cutoff for cross-document boilerplate filtering (default: 0.8)
-t VALUE, --threshold VALUE    Similarity threshold for LexRank graph construction (default: 0.1)
-c VALUE, --convergence VALUE  Convergence tolerance for PageRank graph computation (default: 0.001)
```

### File Formats

The summarizer expects tab-separated text files with each document on a single line.  Each line should contain a document identifier in the first column and the document text in the second column.

Outputs are formatted similarly, but with the text of a single sentence in the second column.

Stopwords are provided as a text file with one word per line.