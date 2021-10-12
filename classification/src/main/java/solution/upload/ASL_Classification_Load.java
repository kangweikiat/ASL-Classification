package solution.upload;

import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class ASL_Classification_Load extends Application {

    public static void main(String[] args) {

        //Load Trained Model



        launch();
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Hello World");
        Group root = new Group();
        Scene scene = new Scene(root,800, 500);
        scene.setFill(Color.ANTIQUEWHITE);

        primaryStage.setTitle("ASL Application");
        primaryStage.setScene(scene);
        primaryStage.show();
    }
}
