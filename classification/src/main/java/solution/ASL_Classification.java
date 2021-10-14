package solution;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.io.File;
import java.io.IOException;

public class ASL_Classification {
    public static void main(String[] args) throws IOException {

        File locationToLoad = new File("asl_trainedmodel.zip");

        Pair<DataSetIterator, DataSetIterator> traintestiter;

        if (!locationToLoad.exists()) {
            System.out.println("Your Model needs training.");
            System.out.println("Training your model.....");
            traintestiter = ASL_Classification_Training.DataRetriever();
            ASL_Classification_Training.networkModel(traintestiter);
            System.out.println("Model Training complete.");
        }

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToLoad);

        traintestiter = ASL_Classification_Training.DataRetriever();

        Evaluation evalTrain = model.evaluate(traintestiter.getKey());
        Evaluation evalTest = model.evaluate(traintestiter.getValue());

        System.out.println("+++++ Evaluation Testing +++++");
        System.out.println("Train evaluation: " + evalTrain.stats());
        System.out.println("Test Evaluation: " + evalTest.stats());
    }
}
