/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package classificator;

import java.io.File;
import java.util.Map;
import java.util.Random;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KDtreeKNN;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.evaluation.CrossValidation;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.featureselection.scoring.GainRatio;
import net.sf.javaml.tools.data.FileHandler;
import net.sf.javaml.tools.weka.WekaClassifier;
import weka.classifiers.functions.SMO;

public class Classificator {

    private static final String DATASET = "/export/Development/DataMining/javaml-0.1.7/UCI-small/iris/iris.data";

    /**
     * Shows the default usage of the KNN algorithm.
     */
    public static void main(String[] args) throws Exception {

        /* Load a data set */
        Dataset data = FileHandler.loadDataset(new File(DATASET), 4, ",");
        /*
         * Contruct a KNN classifier that uses 5 neighbors to make a decision.
         */
        Classifier knn = new KNearestNeighbors(5);
        knn.buildClassifier(data);

        Classifier kdtKnn = new KDtreeKNN(5);
        kdtKnn.buildClassifier(data);


        /*
         * Load a data set for evaluation, this can be a different one, but we
         * will use the same one.
         */
        Dataset dataForClassification = FileHandler.loadDataset(new File(DATASET), 4, ",");
        /* Counters for correct and wrong predictions. */
        int correct = 0, wrong = 0;
        /* Classify all instances and check with the correct class values */
        for (Instance inst : dataForClassification) {
            Object predictedClassValue = knn.classify(inst);
            Object realClassValue = inst.classValue();
            //System.out.println("predicted=" + predictedClassValue.toString() + " real="+realClassValue.toString());
            if (predictedClassValue.equals(realClassValue)) {
                correct++;
            } else {
                wrong++;
            }
        }
        System.out.println("Correct predictions  " + correct);
        System.out.println("Wrong predictions " + wrong);

        /* Performance 
         * 
         */
        System.out.println("Performance ...");
        Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(knn, dataForClassification);
        printPerfMeasure(pm);
        /* 
         * Cross validation
         */
        System.out.println("Cross validation ...");
        /* Construct new cross validation instance with the KNN classifier, */
        CrossValidation cv = new CrossValidation(knn);
        /* 5-fold CV with fixed random generator */
        Map<Object, PerformanceMeasure> p0 = cv.crossValidation(data, 5, new Random(1));
        Map<Object, PerformanceMeasure> p1 = cv.crossValidation(data, 5, new Random(1));
        Map<Object, PerformanceMeasure> p2 = cv.crossValidation(data, 5, new Random(25));
        printPerfMeasure(p0);
        printPerfMeasure(p1);
        printPerfMeasure(p2);

        /*
         * Create Weka classifier 
         */
        System.out.println("Weka classifier ...");
        SMO smo = new SMO();
        /* Wrap Weka classifier in bridge */
        Classifier javamlsmo = new WekaClassifier(smo);
        /* Initialize cross-validation */
        CrossValidation wekaCV = new CrossValidation(javamlsmo);
        /* Perform cross-validation */
        Map<Object, PerformanceMeasure> wekaPm = wekaCV.crossValidation(data);
        /* Output results
         * see http://en.wikipedia.org/wiki/Precision_and_recall
         */
        System.out.println("see http://en.wikipedia.org/wiki/Precision_and_recall" + wekaPm);
        printPerfMeasure(wekaPm);

        /*
         * Feature scoring
         */
        System.out.println("Feature scoring ");
        GainRatio ga = new GainRatio();
        /* Apply the algorithm to the data set */
        ga.build(data);
        /* Print out the score of each attribute */
        for (int i = 0; i < ga.noAttributes(); i++) {
            System.out.println("Attribute[" + i + "] relevance" + ga.score(i));
        }
    }

    private static void printPerfMeasure(Map<Object, PerformanceMeasure> pm) {
        for (Object o : pm.keySet()) {
            System.out.println(o + ": " + pm.get(o).getAccuracy());
        }
    }
}