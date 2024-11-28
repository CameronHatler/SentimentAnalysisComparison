package nlp.ml;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


/**
 * A Naive Bayes model for classifieing a rating as either positive or negative
 * 
 * @author Cameron Hatler
 */
class NBClassifier{

    // instantiate variables, and hashmaps to keep track of counts
    private static final int NEGATIVE = 0;
    private static final int POSITIVE = 1;
    private static final String NEG_LABEL = "0";
    private static final String POS_LABEL = "4";
    private double[] labelProbs = new double[2];  // probability of a positive or negative label
    private ArrayList<HashMap<String, Double>> wordProbs = new ArrayList<>();  // probability of word showing up given label
    private double[] ifNoOccurance = new double[2];  // just the add lambda probability of something that was never seen
    private HashSet<String> vocab; // set containing all words seen in test data.
    

    /**
     * Constructor for NBClassifier, takes in training data with labels
     * and trains the probability of word counts based on Maximum Likelihood Estimation
     * 
     * @param fileIn  the file path of the training data, should be of the form "(positive|negative)[tab]words"
    */
    public NBClassifier(double lambda, String fileIn){

        // initialize probability maps
        labelProbs[NEGATIVE] = 0.0;
        labelProbs[POSITIVE] = 0.0;
        wordProbs.add(NEGATIVE, new HashMap<>());
        wordProbs.add(POSITIVE, new HashMap<>());
        
        // frequency of words given label
        ArrayList<HashMap<String, Integer>> wordCounts = new ArrayList<>(3);  
        wordCounts.add(NEGATIVE, new HashMap<>());
        wordCounts.add(POSITIVE, new HashMap<>());

        // frequency of labels in training data
        int[] labelCounts = new int[2];  
        labelCounts[NEGATIVE] = 0;
        labelCounts[POSITIVE] = 0;

        // frequency of all words given a label
        int[] numWords = new int[2];
        numWords[NEGATIVE] = 0;
        numWords[POSITIVE] = 0;

        // number of unique words in the training set
        vocab = new HashSet<>();

        // read in the training data
        try{

            BufferedReader br = new BufferedReader(new FileReader(fileIn));
            String line;  // a line from training data

            // read through entire file
            while ((line = br.readLine()) != null){
                String[] labelWords = line.split("\",\"");  // seperate the label from rest of words
                if(labelWords.length != 6){
                    System.out.println("incorrect file format, skipping line");
                    continue;
                }
                labelWords[0] = labelWords[0].substring(1);
                labelWords[5] = labelWords[5].substring(0,labelWords[5].length() - 1);
                String[] words = labelWords[5].split("\\s");  // seperate words

                // check whether the label is positive or negative
                int label; // 0 is negative label, 1 is positive label
                if (labelWords[0].equals(NEG_LABEL)){
                    labelCounts[NEGATIVE]++;  // increase negative label counter
                    label = NEGATIVE;
                } else if (labelWords[0].equals(POS_LABEL)){
                    labelCounts[POSITIVE]++;  // increase positive label counter
                    label = POSITIVE;
                } else {
                    continue;
                }

                // add the counts of each word given the label
                for(String word : words){
                    vocab.add(word);
                    wordCounts.get(label).put(word, wordCounts.get(label).getOrDefault(word, 0) + 1);
                    numWords[label]++;
                }
            }

            br.close();  // close the file

            // from the counts, find the probabilities
            for(int label = 0; label <= 1; label++){

                // get the probability of the label
                labelProbs[label] = (double) labelCounts[label] / (double) (labelCounts[NEGATIVE] + labelCounts[POSITIVE]);

                // get the probability of the words given a label
                for(String word : wordCounts.get(label).keySet()){
                    wordProbs.get(label).put(word, (wordCounts.get(label).get(word) + lambda) / (numWords[label] + (vocab.size() * lambda)));
                }

                // the lambda probability if a word is in the vocab, but doesn't occur for a given label
                ifNoOccurance[label] = lambda / (numWords[label] + (vocab.size() * lambda));
            }
        } catch (IOException e){
            e.printStackTrace();
        } catch (Exception e){
            e.printStackTrace();
        }
    }

    
    /**
     * Tests the model on a csv file containing
     * 6 columns "sentiment","ID","data","query","user","tweet"
     * 
     * @param fileIn - the directory to the file
     */
    public void test(String fileIn){

        // keeps track of correct and incorrect guesses
        int correctPosCount = 0;
        int correctNegCount = 0;
        int incorrectPosCount = 0;
        int incorrectNegCount = 0;
        int total = 0;

        try{
            BufferedReader br = new BufferedReader(new FileReader(fileIn));
            String line;
            while ((line = br.readLine()) != null){

                // data is seperated by , with each column wrapped in quotes
                String[] columns = line.split("\",\"");
                if (columns.length != 6) continue;

                // remove quotes at beginning and end of data
                columns[0] = columns[0].substring(1);
                columns[5] = columns[5].substring(0, columns[5].length() - 1);
                if (columns[0].equals("2")) continue;  // skip neutral sentiment
                String[] words = columns[5].split("\\s");

                total++;
                String classification = sentenceClassification(words);
                if (classification.equals(POS_LABEL)){
                    if(classification.equals(columns[0])){
                        correctPosCount++;
                    } else {
                        incorrectPosCount++;
                    }
                } else {
                    if(classification.equals(columns[0])){
                        correctNegCount++;
                    } else {
                        incorrectNegCount++;
                    }
                }
            }
            br.close();
            System.out.println("True Positives: " + correctPosCount);
            System.out.println("True Negatives: " + correctNegCount);
            System.out.println("False Positives: " + incorrectPosCount);
            System.out.println("False Negatives: " + incorrectNegCount);
            double accuracy = (double) (correctNegCount + correctPosCount) / (double) total;
            System.out.println("Model has accuracy of " + accuracy);
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    /**
     * from a file of sentences, classify whether each sentence is positive or negative
     * used if trying to classify sentences as positive or negative
     * @param fileIn the file containing the test sentences
     */
    private void fileClassification(String fileIn){
        try {
            BufferedReader br = new BufferedReader(new FileReader(fileIn));
            String line; 
            String[] words;

            while ((line = br.readLine()) != null){

                // break each line based on whitespace into words then store frequency of words
                String[] columns = line.split("\",\"");
                columns[5] = columns[5].substring(0,columns[5].length() - 1);
                words = columns[5].split("\\s");

                // prints the classification of the line
                System.out.println(sentenceClassification(words));
            }
            br.close();
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    
    /**
     * Classifies a sentence as either positive or negative sentiment
     * @param words the word frequency 'vector' of the sentence
     * @return the sentiment followed by the probability of the sentence given that sentiment
     */
    private String sentenceClassification(String[] words){
        // keep track of the negative and positive probabilities
        double probNeg = Math.log10(labelProbs[NEGATIVE]);
        double probPos = Math.log10(labelProbs[POSITIVE]);

        // make a bag of words
        HashMap<String,Integer> bagOfWords = new HashMap<>();
        // for each word, put it in our word frequency hashmap
        for(String word : words){
            bagOfWords.put(word, bagOfWords.getOrDefault(word, 0) + 1);
        }

        // go through each word to calculate it's probability
        for (String word : bagOfWords.keySet()){

            // skip words that didn't appear in training
            if (!vocab.contains(word)) continue;  

            // add logs of the probability of the word times the frequency of the word in the review
            probNeg += bagOfWords.get(word) * Math.log10(wordProbs.get(NEGATIVE).getOrDefault(word, ifNoOccurance[NEGATIVE]));
            probPos += bagOfWords.get(word) * Math.log10(wordProbs.get(POSITIVE).getOrDefault(word, ifNoOccurance[POSITIVE]));
        }

        // return the most probable option for the recommendation
        if (probNeg > probPos){
            return NEG_LABEL;
        } else {
            return POS_LABEL;
        }
    }


    /**
     * prints the most predictive features of a label for this model
     * @param k the number of predictive features returned
     */
    public void mostPredictiveFeatures(int k){

        // set up hashmaps to store the feature importance
        ArrayList<HashMap<String, Double>> featureImportance = new ArrayList<>(2);
        featureImportance.add(new HashMap<>());
        featureImportance.add(new HashMap<>());

        // make sure the word shows up in both descriptions and then find it's importance
        for(String word : wordProbs.get(NEGATIVE).keySet()){
            if (wordProbs.get(POSITIVE).containsKey(word)){
                featureImportance.get(NEGATIVE).put(word, wordProbs.get(NEGATIVE).get(word) / wordProbs.get(POSITIVE).get(word));

                // we could have saved space since the smallest values in NEGATIVE feature importance will be largest values in POSITIVE feature importance
                // but we would have had to set up two priority queues anyways so this is left for simplicity
                featureImportance.get(POSITIVE).put(word, wordProbs.get(POSITIVE).get(word) / wordProbs.get(NEGATIVE).get(word));
            }
        }

        // set up priority queues to find largest values from the features importances for both labels
        PriorityQueue<Map.Entry<String, Double>> negativeOrdering = new PriorityQueue<>((a,b) -> Double.compare(b.getValue(), a.getValue()));
        negativeOrdering.addAll(featureImportance.get(NEGATIVE).entrySet());
        PriorityQueue<Map.Entry<String, Double>> positiveOrdering = new PriorityQueue<>((a,b) -> Double.compare(b.getValue(), a.getValue()));
        positiveOrdering.addAll(featureImportance.get(POSITIVE).entrySet());

        // print out k most important features for negative label
        System.out.println(k + " most predictive features for negative label");
        for (int i = 0; i < k; i++){
            Map.Entry<String, Double> entry = negativeOrdering.poll();
            System.out.println(entry.getKey());
        }

        // print out k most important features for positive label
        System.out.println("\n" + k + " most predictive features for positive label");
        for (int i = 0; i < k; i++){
            Map.Entry<String, Double> entry = positiveOrdering.poll();
            System.out.println(entry.getKey());
        }
    }


    /**
     * Prints all probabilities of the model, starting with
     * p(negative) and p(positive) followed by the probability
     * of each word given the label negative or positive
     */
    public String toString(){
        StringBuffer buffer = new StringBuffer();

        // probabilities of the labels
        buffer.append("p(negative)= " + labelProbs[NEGATIVE] + "\t\t p(positive)= " + labelProbs[POSITIVE]);
        for(String word : vocab){

            // probability of a word given the different labels
            buffer.append("\np(" + word + "|negative)= " + wordProbs.get(NEGATIVE).getOrDefault(word, ifNoOccurance[NEGATIVE]));
            buffer.append("\np(" + word + "|positive)= " + wordProbs.get(POSITIVE).getOrDefault(word, ifNoOccurance[POSITIVE]));
          }
        return buffer.toString();
    }
    

    public static void main(String[] args) {
        try{
            if (args.length != 3) throw new Exception("Incorrect Number of Arguments");
            String data = args[0];  // training data
            String test = args[1];  // test data
            double lambda = Double.parseDouble(args[2]);  // lambda smoothing

            // set up the model and analyze sentiment of input
            NBClassifier model = new NBClassifier(lambda, data);
            model.test(test);
            
            // System.out.println(model);
            // model.mostPredictiveFeatures(10);
        } catch (Exception e){
            e.printStackTrace();
        }
    }
}