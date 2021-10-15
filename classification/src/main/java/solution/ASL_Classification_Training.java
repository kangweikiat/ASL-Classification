package solution;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

class ASL_Classification_Training {

    static Random rng = new Random();
    static String[] allowedFormats = BaseImageLoader.ALLOWED_FORMATS;
    static PathLabelGenerator labelGenerator = new ParentPathLabelGenerator(); //label maker
    static double trainfrac = 0.8;
    static double testfrac = 1 - trainfrac;
    static int labelIndex = 1; // index 1 is used for data labels
    static int batchSize = 30; // can be adjusted for accuracy
    static int noOfEpoch = 5;
    static double learningrate = 1e-3; //learning rate static
    static int noOfChannels = 3; // no of channels = 3 because RGB
    static int numberofclass = 5; // no of class = 5 (a,b,c,del and space)
    static int height = 50; // height = 50
    static int width = 50; // width = 50

    public static Pair<DataSetIterator, DataSetIterator> DataRetriever() throws IOException {
        File allTrainData = new ClassPathResource("asl/asl_alphabet_train").getFile();
        File allTestData = new ClassPathResource("asl/asl_alphabet_test").getFile();

        FileSplit trainSplit = new FileSplit(allTrainData, allowedFormats, rng); //implementation on input split f(x)
        FileSplit testSplit = new FileSplit(allTestData, allowedFormats, rng); //splits up root directory into files
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedFormats, labelGenerator);
        InputSplit[] sample = trainSplit.sample(pathFilter, 0.8, trainfrac, testfrac);
        //InputSplit[] = a list of loadable locations exposed as an iterator.
        InputSplit traindata = sample[0];
        InputSplit testdata = sample[1];

        ImageRecordReader trainrr = new ImageRecordReader(height, width, noOfChannels, labelGenerator);
        ImageRecordReader testrr = new ImageRecordReader(height, width, noOfChannels, labelGenerator);

        ImageTransform hFlip = new FlipImageTransform(1);
        ImageTransform rotate = new RotateImageTransform(15);
        ImageTransform rCrop = new RandomCropTransform(60, 60);

        List<Pair<ImageTransform, Double>> augList = Arrays.asList(
                new Pair<>(hFlip, 0.4),
                new Pair<>(rotate, 0.5),
                new Pair<>(rCrop, 0.3)
        );

        ImageTransform pipeline = new PipelineImageTransform(augList, false);
        trainrr.initialize(traindata, pipeline);
        testrr.initialize(testdata);

        DataSetIterator trainiter = new RecordReaderDataSetIterator(trainrr, batchSize, labelIndex, numberofclass);
        DataSetIterator testiter = new RecordReaderDataSetIterator(testrr, batchSize, labelIndex, numberofclass);

        DataNormalization scaler = new ImagePreProcessingScaler();
        trainiter.setPreProcessor(scaler);
        testiter.setPreProcessor(scaler);

        return new Pair<DataSetIterator, DataSetIterator>(trainiter, testiter);
    }

    public static void networkModel(Pair<DataSetIterator, DataSetIterator> traintestiter) throws IOException {

        MultiLayerConfiguration nnconfig = new NeuralNetConfiguration.Builder() //configuring NN
                //seed = setting the initial state of random number generator to ensure reproducibility.
                .seed(56) //so that codes can be executed again with similar results
                .weightInit(WeightInit.XAVIER) // weight initializer
                // XAVIER scheme = Gaussian distribution with mean 0, variance 2.0/(fanIn + fanOut)
                .updater(new Adam(learningrate)) // updater = to optimize the learning rate
                // until the neural network converges on its most performant state.
                .list() //started to list down all available layers for NN configuration
                .layer(0, new ConvolutionLayer.Builder() //convolution layer
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .nIn(noOfChannels)
                        .nOut(24)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder() //pooling layer
                        .poolingType(SubsamplingLayer.PoolingType.MAX) //max pixel value of a batch is selected
                        //max pixel value is selected while downsampling
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nOut(12) //no of output neurons of hidden layer
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder()
                        .nOut(numberofclass)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        //multiclass cross entropy - one hot(no rank)
                        .activation(Activation.SOFTMAX)
                        .build())

                .setInputType(InputType.convolutional(height, width, noOfChannels)) //CNN input
                .build(); //end of NN configuration

        MultiLayerNetwork model = new MultiLayerNetwork(nnconfig);
        model.init();

        model.setListeners(new ScoreIterationListener(10)); //no need loop because not CSV

        model.fit(traintestiter.getKey(), noOfEpoch);

        File locationToSave = new File("asl_trainedmodel.zip");
        boolean saveUpdater = false;
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
    }
}
