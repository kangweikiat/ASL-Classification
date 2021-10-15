package solution;

import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.net.URL;

public class ASL_Classification_Load extends Application implements EventHandler<ActionEvent> {
    //Left layout
    Label uploadLabel;
    Image image;
    ImageView imageView;
    FileChooser fileChooser;
    Button button;
    VBox vLeft;
    VBox vCenter;
    VBox vRight;

    //center layout;
    Button predictButton;

    //right layout
    Label predictLabel;
    Text text;
    Text outputText;

    HBox hLayout;
    File response;

    private static MultiLayerNetwork model = null;
    private static final int noOfChannels = 3; // no of channels = 3 because RGB
    private static final int height = 50; // height = 50
    private static final int width = 50; // wid

    public static void main(String[] args) throws IOException {
        File locationToLoad = new File("asl_trainedmodel.zip");
        //Load Trained Model
        model = ModelSerializer.restoreMultiLayerNetwork(locationToLoad);
        System.out.println(model.summary());

        launch();
    }

    @Override
    public void start(Stage primaryStage) throws IOException {
        primaryStage.setTitle("ASL Application");
        String cssLayout = "-fx-border-color: black;\n" +
                "-fx-border-insets: 5;\n" +
                "-fx-border-width: 3;\n" +
                "-fx-border-style: solid;\n";

        //Left layout
        URL emptyImageURL = new ClassPathResource("image/empty-icon.png").getURL();
        uploadLabel = new Label("Upload Image");
        uploadLabel.setFont(new Font(25));
        imageView = new ImageView(emptyImageURL.toString());
        button = new Button("Upload");
        button.setMaxSize(200, 50);
        button.setOnAction(this);
        vLeft = new VBox(50);
        vLeft.getChildren().addAll(uploadLabel, imageView, button);
        vLeft.setAlignment(Pos.CENTER);
        vLeft.setPrefWidth(250);
        vLeft.setStyle(cssLayout);

        //Center Layout
        predictButton = new Button("Predict");
        predictButton.setPrefSize(200, 80);
        predictButton.setOnAction(this);
        vCenter = new VBox();
        vCenter.getChildren().add(predictButton);
        vCenter.setAlignment(Pos.CENTER);
        vCenter.setPrefWidth(250);

        //Right Layout
        predictLabel = new Label("Predicted Value");
        predictLabel.setFont(new Font(25));
        VBox border = new VBox();
        text = new Text(null);
        text.setFont(new Font(50));
        outputText = new Text(null);
        border.getChildren().addAll(text,outputText);
        border.setStyle(cssLayout);
        border.setAlignment(Pos.CENTER);
        vRight = new VBox(50);
        vRight.getChildren().addAll(predictLabel, border);
        vRight.setAlignment(Pos.CENTER);
        vRight.setPrefWidth(250);

        hLayout = new HBox();
        hLayout.setSpacing(100);
        hLayout.getChildren().addAll(vLeft, vCenter, vRight);
        hLayout.setAlignment(Pos.CENTER);

        Scene scene = new Scene(hLayout, 1300, 500);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    @Override
    public void handle(ActionEvent actionEvent) {
        if (actionEvent.getSource().equals(button)) {
            fileChooser = new FileChooser();
            configureFileChooser(fileChooser);
            response = fileChooser.showOpenDialog(null);
            if (response != null) {
                image = new Image(response.toURI().toString(), 200, 200, true, false);
                imageView.setImage(image);
                System.out.println("Path:\n" + response.getAbsoluteFile().getAbsolutePath());
                System.out.println(image.getHeight() + " " + image.getWidth());
            }
        }
        if (actionEvent.getSource().equals(predictButton)) {
            // Feed image to trained model
            NativeImageLoader loader = new NativeImageLoader(height, width, noOfChannels, true);
            INDArray imageArray = null;
            try {
                imageArray = loader.asMatrix(response);
            } catch (IOException e) {
                e.printStackTrace();
            }
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(imageArray);
            INDArray output = model.output(imageArray);

            int prediction = model.predict(imageArray)[0];
            String alphabet = null;
            if (prediction == 0){ alphabet = "A";}
            else if(prediction == 1) { alphabet = "B";}
            else if(prediction == 2){alphabet = "C";}
            else if (prediction == 3) {alphabet = "Del";}
            else if(prediction == 4) {alphabet = "Space";}

            System.out.println("Prediction: " + model.predict(imageArray)[0] + "\n " + output);
            text.setText(alphabet);
            outputText.setText(String.valueOf(output));
        }
    }

    private static void configureFileChooser(final FileChooser fileChooser) {
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("All Images", "*.*"),
                new FileChooser.ExtensionFilter("JPG", "*.jpg"),
                new FileChooser.ExtensionFilter("PNG", "*.png")
        );
    }
}
