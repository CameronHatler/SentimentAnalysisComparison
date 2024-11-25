#! /bin/bash

# This is just an example for Java.  You'll need to modify it for your package structure
# and names.  See the last assignment for more examples.
javac nlp/ml/*.java
java -Xmx2G -cp . nlp.ml.NBClassifier $1 $2 $3
