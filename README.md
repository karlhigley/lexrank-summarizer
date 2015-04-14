# LexRank Summarizer

This is a Spark-based extractive summarizer, which selects the 5 most informative sentences from each document in a large corpus.

## Usage

Build a JAR file from the source with `sbt assembly`.  Submit a job to Spark with:

```
spark-submit --class io.github.karlhigley.Summarizer <path to jar file>
```

### Input/Output Format

The summarizer expects tab-separated text files with each document on a single line.  Each line should contain a document identifier in the first column and the document text in the second column. Outputs are formatted similarly, but with the text of a single sentence in the second column.


### Stopwords

Stopwords are provided as a text file with one word per line.


### Paths
Input dir:      ./input
Output dir:     ./output
Stopwords file: ./stopwords